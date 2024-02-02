from processors.utils import load_char_to_ids_dict
from processors.file_utils import is_tf_available
from processors.utils import DataProcessor, InputExample, InputFeatures

if is_tf_available():
    import tensorflow as tf

def emb_convert_examples_to_features(examples, tokenizer, #examples è della classe InputExample
                                    max_length=512,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True,
                                    char_vocab_file="./data/dict/bert_char_vocab",
                                    model_type='bert'):
    
    """
    examples: list of str
    """
    char2ids_dict = load_char_to_ids_dict(char_vocab_file=char_vocab_file)
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    features = []
    for (ex_index, example) in enumerate(examples):
        
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)
        
        #TODO: probabilmente puoi fare in modo che example sia una lista di stringhe e non un oggetto InputExample e poi lo passi direttamente a tokenizer.encode_plus
        inputs = tokenizer.encode_plus(
            example,
            #example.text_a,
            #example.text_b, #useless for encoding
            add_special_tokens=True,
            max_length=max_length,
        )
        input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token] * padding_length) + input_ids
            attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(len(input_ids), max_length)
        assert len(attention_mask) == max_length, "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert len(token_type_ids) == max_length, "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        # add char level information
        all_seq_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        #print(f'all_seq_tokens: {" ".join(all_seq_tokens)}')
        char_ids = []
        start_ids = []
        end_ids = []
        char_maxlen = max_length * 6
        for idx, token in enumerate(all_seq_tokens):
            token = token.strip("##")
            #if token == tokenizer.unk_token and idx in span["token_to_orig_map"]:
            #    token_orig_index = span["token_to_orig_map"][idx]
            #    orig_token = example.doc_tokens[token_orig_index]
            #    print(f'UNK: {token} to orig_tokens: {orig_token}')    
            #    token = orig_token
            if len(char_ids) >= char_maxlen:
                break
            token = token.strip("##")
            if token in [tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token,\
                tokenizer.cls_token, tokenizer.mask_token]:
                start_ids.append(len(char_ids))
                end_ids.append(len(char_ids))
                char_ids.append(0)
            else:
                for char_idx, c in enumerate(token):
                    if len(char_ids) >= char_maxlen:
                        break
                    
                    if char_idx == 0:
                        start_ids.append(len(char_ids))
                    if char_idx == len(token) - 1:
                        end_ids.append(len(char_ids))

                    if c in char2ids_dict:
                        cid = char2ids_dict[c]
                    else:
                        cid = char2ids_dict["<unk>"]
                    char_ids.append(cid)

            if len(char_ids) < char_maxlen:
                char_ids.append(0)
            #print(f'token[{token}]: {" ".join(map(str, char_ids[-1*(len(token)+2):]))}')

        if len(char_ids) > char_maxlen:
            char_ids = char_ids[:char_maxlen]
        else:
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
        while len(start_ids) < max_length:
            start_ids.append(char_maxlen-1)
        while len(end_ids) < max_length:
            end_ids.append(char_maxlen-1)

        if False:
            print(f'char_ids : {" ".join(map(str, char_ids))}')
            print(f'start_ids: {" ".join(map(str, start_ids))}')
            print(f'end_ids  : {" ".join(map(str, end_ids))}')

        #logger tolto per la funzione, puoi convertire in print
        """if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))"""
            #logger.info("label: %s (id = %d)" % (example.label, label))

        features.append(
                InputFeatures(char_input_ids=char_ids,
                                start_ids=start_ids,
                                end_ids=end_ids,
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids))

    if is_tf_available() and is_tf_dataset:
        def gen():
            for ex in features:
                yield ({'input_ids': ex.input_ids,
                        'attention_mask': ex.attention_mask,
                        'token_type_ids': ex.token_type_ids},
                        ex.label)

        return tf.data.Dataset.from_generator(gen,
            ({'input_ids': tf.int32,
            'attention_mask': tf.int32,
            'token_type_ids': tf.int32},
            tf.int64),
            ({'input_ids': tf.TensorShape([None]),
            'attention_mask': tf.TensorShape([None]),
            'token_type_ids': tf.TensorShape([None])},
            tf.TensorShape([])))

    return features