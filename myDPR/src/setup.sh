#!/bin/bash

conda create -n kordpr python==3.8.12
conda activate kordpr
pip install torch==1.10.2
pip install faiss-gpu scipy numpy pandas tqdm transformers sentencepiece pytest wandb
pip install git+https://git@github.com/SKTBrain/KoBERT.git@master
pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
pip install wikiextractor

wget https://dumps.wikimedia.org/kowiki/20240301/kowiki-20240301-pages-articles.xml.bz2
bunzip2 kowiki-20240301-pages-articles.xml.bz2

mkdir dataset & wget https://korquad.github.io/dataset/KorQuAD_v1.0_train.json -O KorQuAD_v1.0_train.json
wget https://korquad.github.io/dataset/KorQuAD_v1.0_dev.json -O KorQuAD_v1.0_dev.json