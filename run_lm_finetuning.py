# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
import copy

from datasets import load_dataset

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
import collections
import nltk as tk
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8')
from time import time

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from modeling.modeling_charbert import CharBertForMaskedLM
from modeling.modeling_roberta import RobertaForMaskedLM
from modeling.configuration_bert import BertConfig
from modeling.configuration_roberta import RobertaConfig

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'bert': (BertConfig, CharBertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
}

WIKIPEDIA_DATASETS = {
    'wikipedia_en': ('wikipedia', "20220301.en", 'train[:10%]'),
    'wikipedia_it': ("wikipedia", "20220301.it", 'train[:50%]')
}

MaskedLmInstance = collections.namedtuple("MaskedLmInstance",["index", "label"])

def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        self.char2ids_dict = self.load_line_to_ids_dict(fname=args.char_vocab)
        self.term2ids_dict = self.load_line_to_ids_dict(fname=args.term_vocab)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, args.data_version + '_cached_lm_' + str(block_size) + '_' + filename)

        file_raws = 0
        with open(file_path, 'r', encoding="utf-8") as f:
            for _ in f:
                file_raws += 1
        self.file_raws = file_raws
        self.nraws = args.input_nraws #numero di righe da leggere ad ogni batch
        self.shuffle = True
        self.file_path = file_path
        self.finput = open(file_path, encoding="utf-8")
        self.current_sample_idx = -1
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.num_nraws = 0
        self.args = args
        self.rng = random.Random(args.seed)

    def read_nraws(self):
        self.num_nraws += 1
        logger.info("Reading the [%d]th data block from dataset file at %s" % (self.num_nraws, self.file_path))

        text = ""
        for _ in range(self.nraws):
            line =  self.finput.readline() #legge una riga del file 
            if line: #se non è vuota la aggiunge al testo
                text += line.strip()
            else:
                self.finput.seek(0)
                line =  self.finput.readline()
                text += line.strip()
                
        doc_tokens = tk.word_tokenize(text)
        if self.args.output_debug:
            print(f"doc_tokens : {' '.join(doc_tokens)}")

        tokenized_tokens = []
        sub_index_to_orig_token = {}
        sub_index_to_change = {}
        adv_labels = []
        num_diff = num_same = 0
        for idx, token in enumerate(doc_tokens):
            ori_token = copy.deepcopy(token)
            if self.rng.random() < self.args.adv_probability:
                token = self.create_adv_word(token, self.rng)
            if ori_token != token and self.args.output_debug:
                if num_diff % 1000 == 0:
                    print(f"Change the token {ori_token} To {token}")
                num_diff += 1
            else:
                num_same += 1
            sub_tokens = []
            if self.args.model_type == 'roberta':
                sub_tokens = self.tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = self.tokenizer.tokenize(token)
            for sub_w in sub_tokens:
                sub_index_to_orig_token[len(tokenized_tokens)] = token
                if ori_token != token:
                    sub_index_to_change[len(tokenized_tokens)] = True
                    #if ori_token in self.term2ids_dict:
                    #    adv_labels.append(self.term2ids_dict[ori_token])
                    if ori_token.lower() in self.term2ids_dict:
                        adv_labels.append(self.term2ids_dict[ori_token.lower()])
                    else:
                        adv_labels.append(self.term2ids_dict['<unk>'])
                else:
                    sub_index_to_change[len(tokenized_tokens)] = False
                    adv_labels.append(-1)
                    
                tokenized_tokens.append(sub_w)
        if self.args.output_debug:
            print(f"num_same: {num_same} num_diff: {num_diff}")
            print(f"tokenized doc: {' '.join(tokenized_tokens)}")

        input_tokens, mask_labels = self.create_masked_lm_predictions(tokenized_tokens,\
                self.args.mlm_probability, self.tokenizer, self.rng, sub_index_to_change)
        tokenized_text = self.tokenizer.convert_tokens_to_ids(input_tokens)

        if self.args.output_debug:
            print(f"mask tokens: {' '.join(input_tokens)}")

        seq_maxlen = self.block_size - 2
        self.examples = []
        for i in range(0, len(tokenized_text)-seq_maxlen+1, seq_maxlen): # Truncate in block of block_size
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+seq_maxlen])
            labels = [-1] + mask_labels[i:i+seq_maxlen] + [-1] #For CLS and SEP
            adv_input_labels = [-1] + adv_labels[i:i+seq_maxlen] + [-1]
            char_input_ids, start_ids, end_ids = self.build_char_inputs(input_ids, sub_index_to_orig_token, i, self.rng, labels)
            assert len(input_ids) == len(labels)
            assert len(input_ids) == len(adv_input_labels)
            assert len(input_ids) == len(start_ids)
            assert len(input_ids) == len(end_ids)
            self.examples.append((torch.tensor(char_input_ids), torch.tensor(start_ids), torch.tensor(end_ids),\
                torch.tensor(input_ids), torch.tensor(labels), torch.tensor(adv_input_labels)))
        self.current_sample_idx = -1
        if self.shuffle:
            random.shuffle(self.examples)

    def create_masked_lm_predictions(self, tokens, masked_lm_prob, tokenizer, rng, sub_index_to_change):
        """Creates the predictions for the masked LM objective."""
        
        vocab_words = list(tokenizer.vocab.keys())
        cand_indexes = []
        for (i, token) in enumerate(tokens):
            if token == "[CLS]" or token == "[SEP]" or token == "[PAD]":
                continue
            # Whole Word Masking means that if we mask all of the wordpieces
            # corresponding to an original word. When a word has been split into
            # WordPieces, the first token does not have any marker and any subsequence
            # tokens are prefixed with ##. So whenever we see the ## token, we
            # append it to the previous set of word indexes.
            #
            # Note that Whole Word Masking does *not* change the training code
            # at all -- we still predict each WordPiece independently, softmaxed
            # over the entire vocabulary.
            if len(cand_indexes) >= 1 and token.startswith("##"):
                cand_indexes[-1].append(i)
            else:
                cand_indexes.append([i])

        rng.shuffle(cand_indexes)

        output_tokens = list(tokens)

        num_to_predict = int(round(len(tokens) * masked_lm_prob))

        masked_lms = []
        covered_indexes = set()
        for index_set in cand_indexes:
            if len(masked_lms) >= num_to_predict:
                break
            # If adding a whole-word mask would exceed the maximum number of
            # predictions, then just skip this candidate.
            if len(masked_lms) + len(index_set) > num_to_predict:
                continue
            is_any_index_covered = False
            for index in index_set:
                if index in covered_indexes or sub_index_to_change[index]:
                    is_any_index_covered = True
                    break
            if is_any_index_covered:
                continue
            for index in index_set:
                covered_indexes.add(index)

                masked_token = None
                # 80% of the time, replace with [MASK]
                if rng.random() < 0.8:
                    masked_token = "[MASK]"
                else:
                    # 10% of the time, keep original
                    if rng.random() < 0.5:
                        masked_token = tokens[index]
                        if self.args.output_debug and False:
                            print(f"Keep the original token: {masked_token}")
                    # 10% of the time, replace with random word
                    else:
                        masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

                output_tokens[index] = masked_token

                masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        assert len(masked_lms) <= num_to_predict
        masked_lms = sorted(masked_lms, key=lambda x: x.index)

        masked_lm_labels = [-1] * len(tokens)
        for p in masked_lms:
            #masked_lm_positions.append(p.index)
            #masked_lm_labels.append(p.label)
            masked_lm_labels[p.index] = tokenizer.convert_tokens_to_ids(p.label)

        return output_tokens, masked_lm_labels

    def create_adv_word(self, orig_token, rng):
        # HERE IS WHERE THE MAGIC HAPPENS
        
        token = list(copy.deepcopy(orig_token))
        if len(orig_token) < 4:
            rand_idx = rng.randint(0, 80)
            rand_char = list(self.char2ids_dict.keys())[rand_idx]
            insert_idx = rng.randint(0, len(orig_token)-1)
            token = token[:insert_idx] + [rand_char] + token[insert_idx:]
            if self.args.output_debug and False:
                print(f"Insert the char:{rand_char} orig_token: {orig_token} new_token: {token}")
        else:
            if rng.random() < 0.5:
                rand_idx = rng.randint(0, len(orig_token)-1)
                del token[rand_idx]
                if self.args.output_debug and False:
                    print(f"Delete the char:{orig_token[rand_idx]} orig_token: {orig_token} new_token: {token}")
            else:
                idx = random.randint(1, len(orig_token)-2)
                token[idx], token[idx+1] = token[idx+1], token[idx]
                if self.args.output_debug and False:
                    print(f"Swap the char:{token[idx:idx+2]} orig_token: {orig_token} new_token: {token}")
        token = ''.join(token)
        return token

    def build_char_inputs(self, input_ids, sub_index_to_ori_tok, start, rng, labels):
        all_seq_tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        #if True:
        if self.args.output_debug:
            print(f"all_seq_tokens: {' '.join(all_seq_tokens)}")
        char_ids = []
        start_ids = []
        end_ids = []
        char_maxlen = self.args.block_size * self.args.char_maxlen_for_word
        for idx, token in enumerate(all_seq_tokens):
            if len(char_ids) >= char_maxlen:
                break
            token = token.strip("##")
            if token == self.tokenizer.unk_token:
                tok_orig_index = idx+start - 1
                if tok_orig_index in sub_index_to_ori_tok : #-1 for CLS
                    orig_token = sub_index_to_ori_tok[tok_orig_index]
                    #print(f'UNK: {token} to orig_tokens: {orig_token}')    
                    token = orig_token
            if token in ["[CLS]", "[SEP]", "[MASK]", "[PAD]"] or labels[idx] != -1:
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

                    if c in self.char2ids_dict:
                        cid = self.char2ids_dict[c]
                    else:
                        cid = self.char2ids_dict["<unk>"]
                    char_ids.append(cid)

            if len(char_ids) < char_maxlen:
                char_ids.append(0)
            #if True:
            if self.args.output_debug:
                print(f'token[{token}]: {" ".join(map(str, char_ids[-1*(len(token)+2):]))}')
        #print(f'len of char_ids: {len(char_ids)}')
        if len(char_ids) > char_maxlen:
            char_ids = char_ids[:char_maxlen]
        else:
            pad_len = char_maxlen - len(char_ids)
            char_ids = char_ids + [0] * pad_len
        while len(start_ids) < self.args.block_size:
            start_ids.append(char_maxlen-1)
        while len(end_ids) < self.args.block_size:
            end_ids.append(char_maxlen-1)
        #if True:
        if self.args.output_debug:
            print(f'char_ids : {" ".join(map(str, char_ids))}')
            print(f'start_ids: {" ".join(map(str, start_ids))}')
            print(f'end_ids  : {" ".join(map(str, end_ids))}')
        #max_start = max(start_ids)
        #max_end   = max(end_ids)
        #if max_start > char_maxlen or max_end > char_maxlen:
        #    print("Error sequence information")
        #    exit(0)
        return char_ids, start_ids, end_ids

    def load_line_to_ids_dict(self, fname):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(fname, "r", encoding="utf-8") as reader:
            chars = reader.readlines()
        for index, char in enumerate(chars):
            char = char.rstrip('\n')
            vocab[char] = index
        return vocab

    def __len__(self):
        return self.file_raws

    def __getitem__(self, item):
        #print(f'type(self): {type(self)}')
        #print(f'current_sample_idx: {self.current_sample_idx}')
        self.current_sample_idx += 1
        #print(f'current_sample_idx: {self.current_sample_idx}')

        if len(self.examples) == 0 or self.current_sample_idx == len(self.examples):
            t_0_read_nraws = time()
            self.read_nraws()
            t_1_read_nraws = time()
            print(f'\n\nOne read_nraws takes {t_1_read_nraws - t_0_read_nraws} seconds')
            #print(f'len self.examples: {len(self.examples)}')
            #print(f'self.examples: {self.examples}')
            self.current_sample_idx += 1
            #print(f'current_sample_idx: {self.current_sample_idx}')

        return self.examples[self.current_sample_idx]

class HuggingFaceDataset(TextDataset):
    def __init__(self, tokenizer, args, dataset, block_size=512):
        self.char2ids_dict = self.load_line_to_ids_dict(fname=args.char_vocab)
        self.term2ids_dict = self.load_line_to_ids_dict(fname=args.term_vocab)
        #path, name, split = WIKIPEDIA_DATASETS[dataset_name]
        #self.dataset = load_dataset(path=path, name=name, split=split)
        self.dataset_text = dataset['text']
        
        file_raws = 0
        for doc in tqdm(self.dataset_text, desc='Counting the dataset raws'):
            file_raws += len(doc.splitlines())
        self.file_raws = file_raws
        self.nraws = args.input_nraws #numero di righe da leggere ogni volta
        self.shuffle = True
        self.current_sample_idx = -1
        self.examples = []
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.num_nraws = 0
        self.args = args
        self.rng = random.Random(args.seed)
        
        # NEW VARIABLES
        self.doc_idx = 0
        self.line_idx = 0
        self.start_line_idx = 0
        self.num_tot_docs = len(self.dataset_text)
        
    def read_nraws(self):
        self.num_nraws += 1
        logger.info(f'Reading the {self.num_nraws}th data block from dataset from huggingface (nraws: {self.nraws})')
        
        text = ""
        read_lines = 0
        
        ################
        while read_lines < self.nraws:
            doc = self.dataset_text[self.doc_idx]
            #doc_len = len(doc.splitlines())
            for rel_line_idx, line in enumerate(doc.splitlines()[self.start_line_idx:]):
                abs_line_idx = self.start_line_idx + rel_line_idx
                if line.strip() != '': #se la linea non è vuota
                    print(f'doc_idx: {self.doc_idx}, line_idx: {abs_line_idx}, line: {line.strip()}')
                    text += line.strip()
                    read_lines += 1
                    
                if read_lines == self.nraws: #se ho letto le righe che mi servono prima di finire il doc
                    self.start_line_idx = abs_line_idx + 1 #riparto dalla riga successiva
                    break
                
            if read_lines < self.nraws: #se ho finito il doc ma non ho ancora letto tutte le righe che mi servono
                self.doc_idx += 1 #vado al doc successivo
                self.start_line_idx = 0 #parto dalla riga 0
                if self.doc_idx == self.num_tot_docs: #se ho letto tutti i doc ma mi servono ancora righe
                    self.doc_idx = 0 #riparto da capo
        ################
        
        doc_tokens = tk.word_tokenize(text)
        if self.args.output_debug:
            print(f'doc_tokens: {" ".join(doc_tokens)}')
        
        tokenized_tokens = []
        sub_index_to_orig_token = {}
        sub_index_to_change = {}
        adv_labels = []
        num_diff = num_same = 0
        for idx, token in enumerate(doc_tokens):
            ori_token = copy.deepcopy(token)
            if self.rng.random() < self.args.adv_probability: #con una certa probabilità faccio adv token
                token = self.create_adv_word(token, self.rng)
            if ori_token != token and self.args.output_debug:
                if num_diff % 1000 == 0:
                    print(f"Change the token {ori_token} To {token}")
                num_diff += 1
            else:
                num_same += 1
            sub_tokens = []
            if self.args.model_type == 'roberta':
                sub_tokens = self.tokenizer.tokenize(token, add_prefix_space=True)
            else:
                sub_tokens = self.tokenizer.tokenize(token)
            for sub_w in sub_tokens:
                sub_index_to_orig_token[len(tokenized_tokens)] = token
                if ori_token != token:
                    sub_index_to_change[len(tokenized_tokens)] = True
                    #if ori_token in self.term2ids_dict:
                    #    adv_labels.append(self.term2ids_dict[ori_token])
                    if ori_token.lower() in self.term2ids_dict:
                        adv_labels.append(self.term2ids_dict[ori_token.lower()])
                    else:
                        adv_labels.append(self.term2ids_dict['<unk>'])
                else:
                    sub_index_to_change[len(tokenized_tokens)] = False
                    adv_labels.append(-1)
                    
                tokenized_tokens.append(sub_w)
        if self.args.output_debug:
            print(f"num_same: {num_same} num_diff: {num_diff}")
            print(f"tokenized doc: {' '.join(tokenized_tokens)}")

        input_tokens, mask_labels = self.create_masked_lm_predictions(tokenized_tokens,\
                self.args.mlm_probability, self.tokenizer, self.rng, sub_index_to_change)
        tokenized_text = self.tokenizer.convert_tokens_to_ids(input_tokens)

        if self.args.output_debug:
            print(f"mask tokens: {' '.join(input_tokens)}")

        seq_maxlen = self.block_size - 2
        self.examples = []
        for i in range(0, len(tokenized_text)-seq_maxlen+1, seq_maxlen): # Truncate in block of block_size
            input_ids = self.tokenizer.build_inputs_with_special_tokens(tokenized_text[i:i+seq_maxlen])
            labels = [-1] + mask_labels[i:i+seq_maxlen] + [-1] #For CLS and SEP
            adv_input_labels = [-1] + adv_labels[i:i+seq_maxlen] + [-1]
            char_input_ids, start_ids, end_ids = self.build_char_inputs(input_ids, sub_index_to_orig_token, i, self.rng, labels)
            assert len(input_ids) == len(labels)
            assert len(input_ids) == len(adv_input_labels)
            assert len(input_ids) == len(start_ids)
            assert len(input_ids) == len(end_ids)
            self.examples.append((torch.tensor(char_input_ids), torch.tensor(start_ids), torch.tensor(end_ids),\
                torch.tensor(input_ids), torch.tensor(labels), torch.tensor(adv_input_labels)))
        self.current_sample_idx = -1
        if self.shuffle:
            random.shuffle(self.examples)

def load_and_cache_examples(args, tokenizer, evaluate=False):
    if os.path.isfile(args.train_data_file):
        dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    else:
        path, name, split = WIKIPEDIA_DATASETS[args.train_data_file]
        load = load_dataset(path=path, name=name, split=split, trust_remote_code=True)
        dataset = HuggingFaceDataset(tokenizer, args, load, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu) #default = 4
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=0)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        #global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            t_0_dataloader = time()
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            char_input_ids, start_ids, end_ids, inputs, labels, adv_labels = batch
            char_input_ids = char_input_ids.to(args.device)
            start_ids = start_ids.to(args.device)
            end_ids = end_ids.to(args.device)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            adv_labels = adv_labels.to(args.device)
            model.train()
            outputs = model(char_input_ids, start_ids, end_ids, inputs, masked_lm_labels=labels, adv_labels=adv_labels)
            if args.output_debug:
                print(f'mask_lm loss: {outputs[0]}')
                print(f'adv_term loss: {outputs[1]}')
            loss = outputs[0] + outputs[1] # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
            t_1_dataloader = time()
            print(f'\n\nOne dataloader loop takes {t_1_dataloader - t_0_dataloader} seconds')
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step

def eval_hit_metrics(args, pred_scores, labels):
    p_at_1 = p_at_5 = simple_mask_num = 0
    batch_size = list(pred_scores.size())[0]
    seq_maxlen = list(pred_scores.size())[1]
    pred_score_arr = np.asarray(pred_scores)
    labels_arr = np.asarray(labels)
    for i in range(batch_size):
        simple_pred_scores = pred_score_arr[i]
        simple_labels = labels_arr[i]
        #find the prediction scores for the mask tokens
        mask_idx = np.where(simple_labels != -1)
        mask_scores = simple_pred_scores[mask_idx]
        true_word_ids = simple_labels[mask_idx]
        if args.output_debug:
            print(f"mask_scores shape: {mask_scores.shape}")
            print(f"labels: {list(simple_labels)}")
            print(f"mask_idx: {list(mask_idx)}")
        #assert len(mask_scores) == len(answers) * args.mask_tokens_num
        #calculate the metric for each whole mask position
        len_of_mask_pos = mask_scores.shape[0]
        #true_words = tokenizer.convert_ids_to_tokens(true_word_ids)
        simple_mask_num += len_of_mask_pos
        for seq_idx in range(len_of_mask_pos):
            score_l = mask_scores[seq_idx]
            #true_wordid = true_word_ids[seq_idx]
            nbest_words_idx = score_l.argsort()[-1*5:][::-1]
            nbest_scores = []
            for word_idx in nbest_words_idx:
                nbest_scores.append(score_l[word_idx])
            #nbest_words = tokenizer.convert_ids_to_tokens(nbest_words_idx)
            #true_token = true_words[seq_idx]
            true_id = true_word_ids[seq_idx]
            #if args.output_debug:
            if True:
                print(f"True_id: {true_id}")
                print(f"Nbest pos ids: {' '.join(map(str, nbest_words_idx))}")
                print(f"Nbest scores: {nbest_scores}")
                #print(f"Nbest wordid: {nbest_words_idx}")
                #calculate the matrics
                if true_id == nbest_words_idx[0]:
                    p_at_1  += 1
                    p_at_5  += 1
                    print(f'Hit top 1')
                elif true_id in nbest_words_idx:
                    p_at_5  += 1
                    print(f'Hit top 5')
                else:
                    print(f"Fail to predict")
                print("")
                token_idx_score = []
    
    return p_at_1, p_at_5, simple_mask_num

def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=1)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    softmax_layer = torch.nn.Softmax(dim=-1)

    hit_1 = hit_5 = mask_num_all = 0
    adv_hit_1 = adv_hit_5 = adv_num_all = 0
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        char_input_ids, start_ids, end_ids, inputs, labels, adv_labels_cpu = batch
        char_input_ids = char_input_ids.to(args.device)
        start_ids = start_ids.to(args.device)
        end_ids = end_ids.to(args.device)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        adv_labels = adv_labels_cpu.to(args.device)

        with torch.no_grad():
            outputs = model(char_input_ids, start_ids, end_ids, inputs, masked_lm_labels=labels, adv_labels=adv_labels)
            lm_loss = outputs[0]
            adv_logits = outputs[3]
            adv_scores = softmax_layer(adv_logits)
            adv_scores = adv_scores.detach().cpu()
            sample_adv_hit_1, sample_adv_hit_5, sample_adv_num = eval_hit_metrics(args,\
                adv_scores, adv_labels_cpu)
            adv_hit_1  += sample_adv_hit_1
            adv_hit_5  += sample_adv_hit_5
            adv_num_all += sample_adv_num
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))
    adv_hit_1 = adv_hit_1 * 1.0 / adv_num_all
    adv_hit_5 = adv_hit_5 * 1.0 / adv_num_all

    result = {
        "perplexity": perplexity,
        "adv_hit_at_1": adv_hit_1,
        "adv_hit_at_5": adv_hit_5,
        "adv_all_num": adv_num_all
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument("--char_vocab", default="./data/dict/bert_char_vocab", type=str,
                        help="char vocab for charBert")
    parser.add_argument("--term_vocab", default="/home/rc/wtma/work2/data/wikipedia/term_vocab", type=str,
                        help="term vocab for charBert")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.10,
                        help="Ratio of tokens to mask for masked language modeling loss")
    parser.add_argument("--adv_probability", type=float, default=0.10,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--data_version", default="", type=str,
                        help="training data version for cached file")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--output_debug", action='store_true',
                        help="Whether to output the debug information.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--char_maxlen_for_word", default=6, type=int,
                        help="Max number of char for each word.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--input_nraws", default=10000, type=int,
                        help="number of lines when read the input data each time.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

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

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

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
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    tk.download('punkt')
    main()
