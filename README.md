<p align="center">
  <br>
    <img src="data\our_logo.jpeg" width="150" />  
  <br>
</p>
 
# CharBERT for IR, NER and sentiment analysis

This repository contains our code, datasets and finetuned models developed by our team Poli2Vec for the project of the DNLP course at PoliTo.</br>
Authors: Mingrino Davide, [Pietropaolo Emanuele](https://github.com/edoppiap), [Martini Martina](https://github.com/martina-martini), [Lungo Vaschetti Jacopo](https://github.com/JacopoLungo)

## Base Models + Finetuned
We primarily provide two models. Here are the download links:

Base
* pre-trained CharBERT based on BERT [charbert-bert-wiki](https://drive.google.com/file/d/1rF5_LbA2qIHuehnNepGmjz4Mu6OqEzYT/view?usp=sharing)    
* pre-trained CharBERT based on RoBERTa [charbert-roberta-wiki](https://drive.google.com/file/d/1tkO7_EH1Px7tXRxNDu6lzr_y8b4Q709f/view?usp=sharing)

Finetuned
* finetuned CharBERT on SST2 Plain. [charbert_SST2_plain](https://drive.google.com/drive/folders/1PzOVnI-xa3QjcAJMEXaphW6vGCtzPK_7?usp=sharing)
* finetuned CharBERT on SST2 Adv. [charbert_SST2_adv](https://drive.google.com/drive/folders/1jUGeQmJs1CtNdosqD3zxZnFlkWkXAwlz?usp=sharing)
* finetuned CharBERT on BioMed NER Plain. [charbert_NER_plain](https://drive.google.com/drive/folders/1pftUQdph0iHDsX0ov1C6zaQyBK8Dkots?usp=sharing)
* finetuned CharBERT on BioMed NER Adv. [charbert_NER_adv](https://drive.google.com/drive/folders/1JIUGWuIti4ve_tS81lboV4wssfM6xpG7?usp=sharing)

## Datasets
### SST2
[SST2 plain](https://drive.google.com/drive/folders/1mkVKV_VB8baqifsxg1WlclbuvQzf_laH?usp=sharing)<br/>
[SST2 Adv.](https://drive.google.com/drive/folders/1Dc3O0Tw8UhVRImASrzycoSfkstKXRJDo?usp=sharing)

### BioMed NER
[BioMed plain](https://drive.google.com/drive/folders/1CymDfyDOsaIE2xlMQsa_uIdKCsrJ8vam?usp=sharing)<br/>
[BioMed Adv.](https://drive.google.com/drive/folders/1lyy5MrRCTdIq6sG-ZsPyAKFuDg0td2eN?usp=sharing)

## Directory Guide
```
root_directory
    |- modeling    # contains source codes of CharBERT model part
    |- data   # Character attack datasets and the dicts for CharBERT
    |- processors # contains source codes for processing the datasets
    |- IR_eval # contains source codes for testing the IR part of our RAG
    |- notebooks # contains our notebooks 
    |- shell     # the examples of shell script for training and evaluation
    |- run_*.py  # codes for pre-training or finetuning

```

## Requirements
```
Python 3.6  
Pytorch 1.2
Transformers 2.4.0
sentence_transformers 2.4.0
```

## Performance

### Sentence Classification (SST2)
#### Plain Train
| Model | Plain test | Adv. test | 
| :------- | :---------: | :---------: 
| BERT  | 92.09 | 89.56 |
| CharBERT  | 91.28 | 89.22 |

#### Adv. Train
| Model | Plain test | Adv. test | 
| :------- | :---------: | :---------: 
| BERT  | 92.20 | 90.02 |
| CharBERT  | 90.94 | 90.25 |

### Token Classification (BioMed NER)
#### Plain Train
| Model | Plain test | Adv. test | 
| :------- | :---------: | :---------: 
| BERT  | 27.62 | 31.61 |
| CharBERT  | 46.01 | 50.09 |

#### Adv. Train
| Model | Plain test | Adv. test | 
| :------- | :---------: | :---------: 
| BERT  | 31.11 | 27.60 |
| CharBERT  | 54.09 | 47.06 |


## Usage
You may use another hyper-parameter set to adapt to your computing device, but it may require further tuning, especially `learning_rate` and `num_train_epoch.`

### SST2 finetuning
```
MODEL_DIR=YOUR_MODEL_PARH/charbert-bert-pretrain 
SST2_DIR=YOUR_DATA_PATH/SST2
OUTPUT_DIR=YOUR_OUTPUT_PATH/SST2 
python run_glue.py \
    --model_name_or_path ${MODEL_DIR} \
    --task_name sst-2\
    --model_type bert \
    --do_train \
    --do_eval \
    --data_dir ${SST2_DIR} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=16 \
    --per_gpu_train_batch_size=16 \
    --char_vocab ./data/dict/bert_char_vocab \
    --learning_rate 3e-5 \
    --save_steps 1000 \
    --num_train_epochs 8.0 \
    --overwrite_output_dir \
    --output_dir ${OUTPUT_DIR}

```

### BioMed NER finetuning
You can use the plain or adv. partition in the BioMed_DIR variable, downloadable from the links before. 
```
MODEL_DIR=YOUR_MODEL_PARH/charbert-bert-pretrain 
BioMed_DIR=YOUR_DATA_PATH/BioMed_NER
OUTPUT_DIR=YOUR_OUTPUT_PATH/BioMed_NER 
python run_ner.py \
    --model_type bert \
    --do_train \
    --do_eval \
    --model_name_or_path ${MODEL_DIR} \
    --char_vocab ./data/dict/bert_char_vocab \
    --data_dir ${BioMed_DIR} \
    --labels ${BioMed_DIR}/ner_tags.txt \
    --output_dir ${OUTPUT_DIR} \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --per_gpu_train_batch_size 4 \
    --per_gpu_eval_batch_size 4 \
    --save_steps 2000 \
    --max_seq_length 512 \
    --overwrite_output_dir
```