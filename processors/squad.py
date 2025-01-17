from tqdm import tqdm
import collections
import logging
import os
import json
import numpy as np
import sys
import io
from .utils import load_char_to_ids_dict
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

from .tokenization_bert import BasicTokenizer, whitespace_tokenize
from .utils import DataProcessor, InputExample, InputFeatures
from .file_utils import is_tf_available, is_torch_available

if is_torch_available():
    import torch
    from torch.utils.data import TensorDataset

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start : (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _new_check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""
    # if len(doc_spans) == 1:
    # return True
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span["start"] + doc_span["length"] - 1
        if position < doc_span["start"]:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span["start"]
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span["length"]
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def squad_convert_examples_to_features(
    examples, tokenizer, max_seq_length, doc_stride, max_query_length, is_training,\
    return_dataset=False, char_vocab_file="./data/dict/bert_char_vocab", model_type='bert'
):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        doc_stride: The stride used when the context is too large and is split across several features.
        max_query_length: The maximum length of the query.
        is_training: whether to create features for model evaluation or model training.
        return_dataset: Default False. Either 'pt' or 'tf'.
            if 'pt': returns a torch.data.TensorDataset,
            if 'tf': returns a tf.data.Dataset

    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features( 
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    # Defining helper methods
    char2ids_dict = load_char_to_ids_dict(char_vocab_file=char_vocab_file)
    unique_id = 1000000000

    features = []
    #* per ogni oggetto SquadExample
    for (example_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):
        #if example_index == 100:
        #    break
        if is_training and not example.is_impossible:
            # Get start and end position
            start_position = example.start_position #l'indice della parola iniziale della risposta all'interno del contest_text
            end_position = example.end_position # l'indice dell'ultima parola della risposta all'interno del contest_text

            # If the answer cannot be found in the text, then skip this example.
            actual_text = " ".join(example.doc_tokens[start_position : (end_position + 1)]) #stringa della risposta
            cleaned_answer_text = " ".join(whitespace_tokenize(example.answer_text)) #stringa della risposta pulita
            if actual_text.find(cleaned_answer_text) == -1:
                logger.warning("Could not find answer: '%s' vs. '%s'", actual_text, cleaned_answer_text)
                continue

        tok_to_orig_index = [] #lista di indici delle parole del contesto (un indice può comparire più di una volta perchè le parole possono essere divise in più sotto-token)
        orig_to_tok_index = [] #[0, 1, 3]
        #Queste due liste sopra sono utili per mappare gli indici dei sotto-token con gli indici delle parole del contesto e viceversa
        #se passi l'indice di una parola a orig_to_tok_index ti ritorna l'indice del primo sotto-token della parola
        #se passi l'indice di un sotto-token a tok_to_orig_index ti ritorna l'indice della parola a cui appartiene
        #Esempio:
        #               0    1   2
        #  contesto: "ciao come stai"
        #                 0     0    0      1      2     2
        #  sotto-token: ['c', 'ia', 'o', 'come', 'sta', 'i']
        #                 0     1    2      3      4     5

        #tok_to_orig_index = [0, 0, 0, 1, 2, 2]
        #orig_to_tok_index = [0, 3, 4]
        #
        all_doc_tokens = [] #una lista di sotto-token (una parola può essere divisa in più sotto-token) (sono stringhe)
        for (i, token) in enumerate(example.doc_tokens):#per ogni parola del contesto (lista di stringhe)
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = []
            if model_type == 'roberta':
                sub_tokens = tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = tokenizer.tokenize(token) #per ogni parola del contesto, la tokenizza (un token può essere diviso in più sotto-token)
            for sub_token in sub_tokens: #ogni sotto-token viene aggiunto alla lista di sotto-token e viene salvato l'indice della parola a cui appartiene
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer, example.answer_text
            )

        spans = []
        truncated_query = []

        """logger.info(f'tok_to_orig_index: {tok_to_orig_index[:20]}')
        logger.info(f'orig_to_tok_index: {orig_to_tok_index[:20]}')"""


        if model_type == 'roberta':
            truncated_query = tokenizer.encode(
                example.question_text, add_special_tokens=False, max_length=max_query_length, add_prefix_space=True)
        else:
            truncated_query = tokenizer.encode(
                example.question_text, add_special_tokens=False, max_length=max_query_length)
        sequence_added_tokens = tokenizer.model_max_length - tokenizer.max_len_single_sentence
        sequence_pair_added_tokens = tokenizer.model_max_length - tokenizer.max_len_sentences_pair

        span_doc_tokens = all_doc_tokens
        #logger.info("\n\n\n")
        jack = 0
        while len(spans) * doc_stride < len(all_doc_tokens):
            """logger.info(f'\n\ni: {jack}')
            jack += 1
            logger.info(f'len(spans): {len(spans)}')
            logger.info(f'doc_stride: {doc_stride}')
            logger.info(f'len(all_doc_tokens): {len(all_doc_tokens)}')

            logger.info(f'\n\ndoc_stride: {doc_stride}')
            logger.info(f'max_seq_length: {max_seq_length}')
            logger.info(f'len(truncated_query): {len(truncated_query) }')
            logger.info(f'sequence_pair_added_tokens: {sequence_pair_added_tokens}')
            logger.info(f'max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens: {max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens}')

            logger.info(f'len(truncated_query): {len(truncated_query)}')
            logger.info(f'\ntruncated_query_from id to tokens: {tokenizer.convert_ids_to_tokens(truncated_query)}')

            logger.info(f'len(span_doc_tokens): {len(span_doc_tokens)}')
            logger.info(f'span_doc_tokens[:30]: {span_doc_tokens[:30]}')"""

            #tokenizer.padding_side = right
            encoded_dict = tokenizer.encode_plus(
                truncated_query if tokenizer.padding_side == "right" else span_doc_tokens, 
                span_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
                max_length=max_seq_length,
                return_overflowing_tokens=True,
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation="only_second" if tokenizer.padding_side == "right" else "only_first",
                add_prefix_space=True if model_type == 'roberta' else False
            )
            #l'errore è che span_doc_tokens è una lista vuota al secondo giro
            #
            # text: è una lista di tokens ids (è una lista di interi) della query
            # text_pair: è una lista di stringhe che contiene tutti i sotto-token del contesto
            encoded_dict = tokenizer.encode_plus(
                text = truncated_query,
                text_pair = span_doc_tokens,
                max_length=max_seq_length, #384
                return_overflowing_tokens=True, #restituisce i token oltre il max_length
                pad_to_max_length=True,
                stride=max_seq_length - doc_stride - len(truncated_query) - sequence_pair_added_tokens,
                truncation="only_second",
                add_prefix_space= False
            )


            paragraph_len = min(
                len(all_doc_tokens) - len(spans) * doc_stride,
                max_seq_length - len(truncated_query) - sequence_pair_added_tokens,
            )

            #prendi solo gli ids che non sono di padding
            if tokenizer.pad_token_id in encoded_dict["input_ids"]:
                non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
            else:
                non_padded_ids = encoded_dict["input_ids"]

            #tokens contiene i token sia della query che del contesto
            tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

            token_to_orig_map = {}
            for i in range(paragraph_len):
                index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
                token_to_orig_map[index] = tok_to_orig_index[len(spans) * doc_stride + i]

            encoded_dict["paragraph_len"] = paragraph_len
            encoded_dict["tokens"] = tokens
            encoded_dict["token_to_orig_map"] = token_to_orig_map
            encoded_dict["truncated_query_with_special_tokens_length"] = len(truncated_query) + sequence_added_tokens
            encoded_dict["token_is_max_context"] = {}
            encoded_dict["start"] = len(spans) * doc_stride
            encoded_dict["length"] = paragraph_len

            spans.append(encoded_dict)

            if "overflowing_tokens" not in encoded_dict or encoded_dict['overflowing_tokens'] == []:
                break
            span_doc_tokens = encoded_dict["overflowing_tokens"]

        for doc_span_index in range(len(spans)):
            for j in range(spans[doc_span_index]["paragraph_len"]):
                is_max_context = _new_check_is_max_context(spans, doc_span_index, doc_span_index * doc_stride + j)
                index = (
                    j
                    if tokenizer.padding_side == "left"
                    else spans[doc_span_index]["truncated_query_with_special_tokens_length"] + j
                )
                spans[doc_span_index]["token_is_max_context"][index] = is_max_context

        for span in spans:
            # Identify the position of the CLS token
            cls_index = span["input_ids"].index(tokenizer.cls_token_id)

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = np.array(span["token_type_ids"])

            p_mask = np.minimum(p_mask, 1)

            if tokenizer.padding_side == "right":
                # Limit positive values to one
                p_mask = 1 - p_mask

            p_mask[np.where(np.array(span["input_ids"]) == tokenizer.sep_token_id)[0]] = 1

            # Set the CLS index to '0'
            p_mask[cls_index] = 0

            span_is_impossible = example.is_impossible
            start_position = 0
            end_position = 0
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = span["start"]
                doc_end = span["start"] + span["length"] - 1
                out_of_span = False

                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True

                if out_of_span:
                    start_position = cls_index
                    end_position = cls_index
                    span_is_impossible = True
                else:
                    if tokenizer.padding_side == "left":
                        doc_offset = 0
                    else:
                        doc_offset = len(truncated_query) + sequence_added_tokens

                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            # add char level information
            all_seq_tokens = tokenizer.convert_ids_to_tokens(span["input_ids"])
            #print(f'all_seq_tokens: {" ".join(all_seq_tokens)}')
            char_ids = []
            start_ids = []
            end_ids = []
            char_maxlen = max_seq_length * 6
            for idx, token in enumerate(all_seq_tokens):
                if len(char_ids) >= char_maxlen:
                    break
                token = token.strip("##")
                if token == tokenizer.unk_token and idx in span["token_to_orig_map"]:
                    token_orig_index = span["token_to_orig_map"][idx]
                    orig_token = example.doc_tokens[token_orig_index]
                    #print(f'UNK: {token} to orig_tokens: {orig_token}')    
                    token = orig_token
                if token == tokenizer.pad_token:
                    continue
                
                #if token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"]:
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
                if False:
                    print(f'token[{token}]: {" ".join(map(str, char_ids[-1*(len(token)+2):]))}')

            assert len(char_ids) <=  char_maxlen
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
            while len(start_ids) < max_seq_length:
                start_ids.append(char_maxlen-1)
            while len(end_ids) < max_seq_length:
                end_ids.append(char_maxlen-1)
            assert len(char_ids) == char_maxlen
            assert len(start_ids) == max_seq_length
            assert len(end_ids) == max_seq_length
            if False:
                print(f'char_ids: {" ".join(map(str, char_ids))}')
                print(f'start_ids: {" ".join(map(str, start_ids))}')
                print(f'end_ids: {" ".join(map(str, end_ids))}')

            features.append(
                SquadFeatures(
                    char_ids,
                    start_ids,
                    end_ids,
                    span["input_ids"],
                    span["attention_mask"],
                    span["token_type_ids"],
                    cls_index,
                    p_mask.tolist(),
                    example_index=example_index,
                    unique_id=unique_id,
                    paragraph_len=span["paragraph_len"],
                    token_is_max_context=span["token_is_max_context"],
                    tokens=span["tokens"],
                    token_to_orig_map=span["token_to_orig_map"],
                    start_position=start_position,
                    end_position=end_position,
                )
            )

            unique_id += 1

    if return_dataset == "pt":
        if not is_torch_available():
            raise ImportError("Pytorch must be installed to return a pytorch dataset.")

        # Convert to Tensors and build dataset
        all_char_ids = torch.tensor([f.char_ids for f in features], dtype=torch.long)
        all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
        all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)
        all_p_mask = torch.tensor([f.p_mask for f in features], dtype=torch.float)

        if not is_training:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            dataset = TensorDataset(
              all_char_ids, all_start_ids, all_end_ids, all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index, all_p_mask
            )
        else:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)
            dataset = TensorDataset(
                all_char_ids,
                all_start_ids,
                all_end_ids,
                all_input_ids,
                all_attention_masks,
                all_token_type_ids,
                all_start_positions,
                all_end_positions,
                all_cls_index,
                all_p_mask,
            )

        return features, dataset
    elif return_dataset == "tf":
        if not is_tf_available():
            raise ImportError("TensorFlow must be installed to return a TensorFlow dataset.")

        def gen():
            for ex in features:
                yield (
                    {
                        "input_ids": ex.input_ids,
                        "attention_mask": ex.attention_mask,
                        "token_type_ids": ex.token_type_ids,
                    }, {
                        "start_position": ex.start_position,
                        "end_position": ex.end_position,
                        "cls_index": ex.cls_index,
                        "p_mask": ex.p_mask,
                    }
                )

        return tf.data.Dataset.from_generator(
            gen,
            (
                {"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32},
                {"start_position": tf.int64, "end_position": tf.int64, "cls_index": tf.int64, "p_mask": tf.int32},
            ),
            (
                {
                    "input_ids": tf.TensorShape([None]),
                    "attention_mask": tf.TensorShape([None]),
                    "token_type_ids": tf.TensorShape([None]),
                },
                {
                    "start_position": tf.TensorShape([]),
                    "end_position": tf.TensorShape([]),
                    "cls_index": tf.TensorShape([]),
                    "p_mask": tf.TensorShape([None]),
                },
            ),
        )

    return features


class SquadProcessor(DataProcessor):
    """
    Processor for the SQuAD data set.
    Overriden by SquadV1Processor and SquadV2Processor, used by the version 1.1 and version 2.0 of SQuAD, respectively.
    """

    train_file = None
    dev_file = None

    def _get_example_from_tensor_dict(self, tensor_dict, evaluate=False):
        if not evaluate:
            answer = tensor_dict["answers"]["text"][0].numpy().decode("utf-8")
            answer_start = tensor_dict["answers"]["answer_start"][0].numpy()
            answers = []
        else:
            answers = [
                {"answer_start": start.numpy(), "text": text.numpy().decode("utf-8")}
                for start, text in zip(tensor_dict["answers"]["answer_start"], tensor_dict["answers"]["text"])
            ]

            answer = None
            answer_start = None

        return SquadExample(
            qas_id=tensor_dict["id"].numpy().decode("utf-8"),
            question_text=tensor_dict["question"].numpy().decode("utf-8"),
            context_text=tensor_dict["context"].numpy().decode("utf-8"),
            answer_text=answer,
            start_position_character=answer_start,
            title=tensor_dict["title"].numpy().decode("utf-8"),
            answers=answers,
        )

    def get_examples_from_dataset(self, dataset, evaluate=False):
        """
        Creates a list of :class:`~transformers.data.processors.squad.SquadExample` using a TFDS dataset.

        Args:
            dataset: The tfds dataset loaded from `tensorflow_datasets.load("squad")`
            evaluate: boolean specifying if in evaluation mode or in training mode

        Returns:
            List of SquadExample

        Examples::

            import tensorflow_datasets as tfds
            dataset = tfds.load("squad")

            training_examples = get_examples_from_dataset(dataset, evaluate=False)
            evaluation_examples = get_examples_from_dataset(dataset, evaluate=True)
        """

        if evaluate:
            dataset = dataset["validation"]
        else:
            dataset = dataset["train"]

        examples = []
        for tensor_dict in tqdm(dataset):
            examples.append(self._get_example_from_tensor_dict(tensor_dict, evaluate=evaluate))

        return examples

    def get_train_examples(self, data_dir, filename=None):
        """
        Returns the training examples from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the training file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.

        """
        if data_dir is None:
            data_dir = ""

        if self.train_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.train_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "train")

    def get_dev_examples(self, data_dir, filename=None):
        """
        Returns the evaluation example from the data directory.

        Args:
            data_dir: Directory containing the data files used for training and evaluating.
            filename: None by default, specify this if the evaluation file has a different name than the original one
                which is `train-v1.1.json` and `train-v2.0.json` for squad versions 1.1 and 2.0 respectively.
        """
        if data_dir is None:
            data_dir = ""

        if self.dev_file is None:
            raise ValueError("SquadProcessor should be instantiated via SquadV1Processor or SquadV2Processor")

        with open(
            os.path.join(data_dir, self.dev_file if filename is None else filename), "r", encoding="utf-8"
        ) as reader:
            input_data = json.load(reader)["data"]
        return self._create_examples(input_data, "dev")

    def _create_examples(self, input_data, set_type):
        is_training = set_type == "train"
        examples = []
        for entry in tqdm(input_data):
            title = entry["title"]
            for paragraph in entry["paragraphs"]:
                context_text = paragraph["context"]
                for qa in paragraph["qas"]:
                    qas_id = qa["id"]
                    question_text = qa["question"]
                    start_position_character = None
                    answer_text = None
                    answers = []

                    if "is_impossible" in qa:
                        is_impossible = qa["is_impossible"]
                    else:
                        is_impossible = False

                    if not is_impossible:
                        if is_training:
                            answer = qa["answers"][0]
                            answer_text = answer["text"]
                            start_position_character = answer["answer_start"]
                        else:
                            answers = qa["answers"]

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        context_text=context_text,
                        answer_text=answer_text,
                        start_position_character=start_position_character,
                        title=title,
                        is_impossible=is_impossible,
                        answers=answers,
                    )

                    examples.append(example)
        return examples


class SquadV1Processor(SquadProcessor):
    train_file = "train-v1.1.json"
    dev_file = "dev-v1.1.json"


class SquadV2Processor(SquadProcessor):
    train_file = "train-v2.0.json"
    dev_file = "dev-v2.0.json"


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        start_position_character,
        title,
        answers=[],
        is_impossible=False,
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.title = title
        self.is_impossible = is_impossible
        self.answers = answers

        self.start_position, self.end_position = 0, 0

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset

        # Start end end positions only has a value during evaluation.
        if start_position_character is not None and not is_impossible:
            self.start_position = char_to_word_offset[start_position_character]
            self.end_position = char_to_word_offset[
                min(start_position_character + len(answer_text) - 1, len(char_to_word_offset) - 1)
            ]


class SquadFeatures(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        p_mask: Mask identifying tokens that can be answers vs. tokens that cannot.
            Mask with 1 for tokens than cannot be in the answer and 0 for token that can be in an answer
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        token_is_max_context: List of booleans identifying which tokens have their maximum context in this feature object.
            If a token does not have their maximum context in this feature object, it means that another feature object
            has more information related to that token and should be prioritized over this feature for that token.
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
        start_position: start of the answer token index 
        end_position: end of the answer token index 
    """

    def __init__(
        self,
        char_ids,
        start_ids,
        end_ids,
        input_ids,
        attention_mask,
        token_type_ids,
        cls_index,
        p_mask,
        example_index,
        unique_id,
        paragraph_len,
        token_is_max_context,
        tokens,
        token_to_orig_map,
        start_position,
        end_position,
    ):
        self.char_ids = char_ids
        self.start_ids = start_ids
        self.end_ids = end_ids
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index
        self.p_mask = p_mask

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.token_is_max_context = token_is_max_context
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map

        self.start_position = start_position
        self.end_position = end_position


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits
