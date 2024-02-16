import nltk as tk

import copy
import random
import collections

import pandas as pd
from pathlib import Path
import os

from tqdm import tqdm
from nltk.tokenize.treebank import TreebankWordDetokenizer
import argparse

def load_line_to_ids_dict(fname):
        """Loads a vocabulary file into a dictionary."""
        vocab = collections.OrderedDict()
        with open(fname, "r", encoding="utf-8") as reader:
            chars = reader.readlines()
        for index, char in enumerate(chars):
            char = char.rstrip('\n')
            vocab[char] = index
        return vocab
    

def create_adv_word(args, orig_token, rng, debug = False):
    # HERE IS WHERE THE MAGIC HAPPENS
    
    char2ids_dict = load_line_to_ids_dict(args.char_vocab) # "./data/dict/bert_char_vocab"
    
    token = list(copy.deepcopy(orig_token))
    #se la parola è più corta di 4 caratteri, aggiungo un carattere a caso
    if len(orig_token) < 4:
        rand_idx = rng.randint(0, 80) #le prime 80 sono caratteri sensati
        rand_char = list(char2ids_dict.keys())[rand_idx] #prendo un carattere a caso
        insert_idx = rng.randint(0, len(orig_token)-1) #prendo un indice a caso tra 0 e la lunghezza della parola
        token = token[:insert_idx] + [rand_char] + token[insert_idx:] #inserisco il carattere tra i due indici
        if debug:
            print(f"Insert the char:{rand_char} orig_token: {orig_token} new_token: {token}")
    
    #se la parola è più lunga di 4 caratteri, scelgo se cancellare o scambiare un carattere
    else:
        if rng.random() < 0.5: #con prob del 50% tolgo un carattere a caso
            rand_idx = rng.randint(0, len(orig_token)-1)
            del token[rand_idx]
            if debug:
                print(f"Delete the char:{orig_token[rand_idx]} orig_token: {orig_token} new_token: {token}")
        else: #con prob del 50% scambio due caratteri a caso
            idx = random.randint(1, len(orig_token)-2)
            token[idx], token[idx+1] = token[idx+1], token[idx]
            if debug:
                print(f"Swap the char:{token[idx:idx+2]} orig_token: {orig_token} new_token: {token}")
    token = ''.join(token)
    return token

def create_adv_ds(args, sentence, rng, debug = False):
    
    num_diff = num_same = 0
    
    doc_tokens = tk.word_tokenize(sentence)
    processed_tokens = []
    for __, token in enumerate(doc_tokens):
        ori_token = copy.deepcopy(token)
        if rng.random() < args.adv_probability: #con probabilità "adv_probability" cambio la parola
            token = create_adv_word(args, token, rng, debug = debug)
            
        if ori_token != token:
            #if num_diff % 1000 == 0:
            #    print(f"Change the token {ori_token} To {token}")
            num_diff += 1
        else:
            num_same += 1
        
        processed_tokens.append(token)
            
    if debug:
        print(f"num_same: {num_same}, num_diff: {num_diff}")
        #print(f"tokenized doc: {' '.join(tokenized_tokens)}")
        
    return TreebankWordDetokenizer().detokenize(processed_tokens), num_same, num_diff

def main():
    parser = argparse.ArgumentParser(description='Create an adversarial dataset')
    parser.add_argument("--file_path", default=None, type=str, required=True,
                        help="Path to the .tsv file to be processed.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the adversarial dataset will be written.")
    parser.add_argument("--char_vocab", default='./data/dict/bert_char_vocab', type=str, required=False,
                        help="path for character vocab file")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--adv_probability", type=float, default=0.10,
                        help="Probability for a word to be modified")
        
    args = parser.parse_args()
    rng = random.Random(args.seed) #seed = 42
    
    file_path = args.file_path if args.file_path.endswith('.tsv') else args.file_path + 'tsv'
    output_dir = Path(args.output_dir)
    in_file_name = str(Path(file_path).parts[-1])
    
    if '\t' not in str(in_file_name):
        out_file_name = Path('adv_' + in_file_name)
    else:
        out_file_name =  Path('adv_' + in_file_name.split('\t')[-1])
        
    if (output_dir/out_file_name).exists() and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(output_dir/out_file_name))
    
    original_df = pd.read_csv(file_path, sep = "\t")
    print("Length of original dataset: ", len(original_df))
    
    num_same = num_diff = 0 #counter num of same and diff tokens (words)
    adv_test_df = original_df.copy(deep=True)
    
    for idx, row in tqdm(adv_test_df.iterrows(), total = len(adv_test_df)):
        adv_test_df.at[idx,'sentence'], tmp_num_sam, tmp_num_diff = create_adv_ds(args, sentence = row['sentence'], rng = rng)
        num_same += tmp_num_sam
        num_diff += tmp_num_diff
    
    tot_word = num_diff + num_same
    print(f"num_same_word: {num_same} ({(num_same / tot_word):.2f}), num_diff: {num_diff} ({(num_diff / tot_word):.2f})")
    
    adv_test_df.to_csv(output_dir/out_file_name, sep = '\t', index = False)

if __name__ == "__main__":
    tk.download('punkt')
    main()
