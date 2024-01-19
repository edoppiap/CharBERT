from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import json

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME, BertTokenizer, RobertaTokenizer

from modeling.configuration_bert import BertConfig
from modeling.configuration_roberta import RobertaConfig

#from transformers import glue_processors

#from transformers import glue_convert_examples_to_features as convert_examples_to_features
from modeling.modeling_charbert import CharBertModel
from modeling.modeling_roberta import RobertaModel


import logging
import os
import collections
import sys
import io
from processors.utils import load_char_to_ids_dict
#sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')

from processors.utils import DataProcessor, InputExample, InputFeatures
from processors.file_utils import is_tf_available

if is_tf_available():
    import tensorflow as tf

logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, CharBertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
def create_embeddings(args, model, tokenizer):
    results = []
    
    embed_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)
    
    """eval_output_dir = args.output_dir
    
    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)"""

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    embed_sampler = SequentialSampler(embed_dataset)
    embed_dataloader = DataLoader(embed_dataset, sampler=embed_sampler, batch_size=args.eval_batch_size)

    # multi-gpu eval
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(embed_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    for batch in tqdm(embed_dataloader, desc="Creating embeddings"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {'char_input_ids': batch[0],
                        'start_ids':     batch[1],
                        'end_ids':     batch[2],
                        'input_ids':     batch[3],
                        'attention_mask': batch[4]}
            if args.model_type != 'distilbert':
                inputs['token_type_ids'] = batch[5] if args.model_type in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids
            print(f"inputs['start_ids']: {inputs['start_ids']}")
            outputs = model(**inputs) #model deve essere CharBertModel
            
            #sequence_output, pooled_output, hidden_states, attentions = outputs #da spostare sulla gpu
        
        #TODO: bisogna usare questo per avere una rappresentazione della frase con un unico vettore
        #Per avere per ogni frase un unico vettore
        token_seq_repr = outputs[0]
        char_seq_repr = outputs[2]
        print(f'token_seq_repr shape: {token_seq_repr.shape}')
        seq_repr = torch.cat([token_seq_repr, char_seq_repr], dim=-1)
        print(f'seq_repr shape: {seq_repr.shape}')
        seq_output = torch.mean(seq_repr, dim=1)
        results.append(seq_output)
        
        #results.append(outputs)

    return results

def create_examples(lines, set_type) -> InputExample: #TODO fare accettare una cartella contenente file python
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def create_examples_from_file(file_path) -> InputExample:
    #TODO qui in qualche modo deve essere definito il blocco che deve leggere insieme attualmente è singola riga
    examples = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            guid = str(i)
            text_a = line.strip()
            examples.append(InputExample(guid=guid, text_a=text_a))
    return examples
    
def emb_convert_examples_to_features(examples, tokenizer, #examples è della classe InputExample
                                    max_length=512,
                                    pad_on_left=False,
                                    pad_token=0,
                                    pad_token_segment_id=0,
                                    mask_padding_with_zero=True,
                                    char_vocab_file="./data/dict/bert_char_vocab",
                                    model_type='bert'):
    
    char2ids_dict = load_char_to_ids_dict(char_vocab_file=char_vocab_file)
    is_tf_dataset = False
    if is_tf_available() and isinstance(examples, tf.data.Dataset):
        is_tf_dataset = True

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d" % (ex_index))
        if is_tf_dataset:
            example = processor.get_example_from_tensor_dict(example)
            example = processor.tfds_map(example)

        inputs = tokenizer.encode_plus(
            example.text_a,
            example.text_b,
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
        print(f'all_seq_tokens: {" ".join(all_seq_tokens)}')
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

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("start_ids: %s" % " ".join([str(x) for x in start_ids]))
            logger.info("end_ids: %s" % " ".join([str(x) for x in end_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
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

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    """cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        'dev' if evaluate else 'train',
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))"""
    #if os.path.exists(cached_features_file) and not args.overwrite_cache:

    logger.info("Creating embedding from file at %s", args.data_dir)
    
    examples =  create_examples_from_file(file_path = args.data_dir)
    print(f"Begin to convert_examples_to_features...")
    features = emb_convert_examples_to_features(examples, #InputExample
                                            tokenizer,
                                            max_length=args.max_seq_length,
                                            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
                                            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                                            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0,
                                            char_vocab_file=args.char_vocab,
                                            model_type=args.model_type
    )
    """if args.local_rank in [-1, 0]:
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)"""

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_char_ids = torch.tensor([f.char_input_ids for f in features], dtype=torch.long)
    all_start_ids = torch.tensor([f.start_ids for f in features], dtype=torch.long)
    all_end_ids = torch.tensor([f.end_ids for f in features], dtype=torch.long)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_char_ids, all_start_ids, all_end_ids, all_input_ids, all_attention_mask, all_token_type_ids)
    return dataset        

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True, #use this to retrieve the data
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                            "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    #parser.add_argument("--do_train", action='store_true',
    #                    help="Whether to run training.")
    #parser.add_argument("--do_eval", action='store_true',
    #                    help="Whether to run eval on the dev set.")
    #parser.add_argument("--evaluate_during_training", action='store_true',
    #                    help="Rul evaluation during training at each logging step.")

    #parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
    #                    help="Batch size per GPU/CPU for training.")
    """parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")"""

    #parser.add_argument('--save_steps', type=int, default=50,
    #                    help="Save checkpoint every X updates steps.")
    #parser.add_argument("--eval_all_checkpoints", action='store_true',
    #                    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--char_vocab", default='./data/dict/bert_char_vocab', type=str, required=True,
                        help="path for character vocab file")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                            "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '\t %(levelname)s - %(name)s -   %(message)s',
                        #datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    print(f'{config_class =}, {model_class =}, {tokenizer_class =}')
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    logger.info("model_type: %s", type(model))
    
    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Embed parameters %s", args)

    # Create embeddings
    results = {}
    if args.local_rank in [-1, 0]:
        results = create_embeddings(args, model, tokenizer)
    print('result len: ', len(results))
    print('result[0] len:', len(results[0]))
    print('result[0][0] shape:', results[0][0].shape)
    print('result[0][1] shape:', results[0][1].shape)
    print('result[0][2] shape:', results[0][2].shape)
    print('result[0][3] shape:', results[0][3].shape)
    print('result[0][4] shape:', results[0][4].shape)

    #print(f'{results =}')
    return results


if __name__ == "__main__":
    main()