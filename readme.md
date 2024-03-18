# Retrieval

문서 검색 QA task

## Setup

### Install
```python
git clone
conda create -n exper python==3.10
conda activate exper
pip install -r requirements.txt
```

* Reload the session!

## Data

private

## Run

### Chunking & Vector mapping
* BGE, KorSIM, Openai(ada)
  * Openai has only "txt" file type
```python
bash create_index.sh
```

### Run demo
```python
cd src
bash run_demo.sh
```

### Show demo
```python
cat demo.log
```

### Evaluation
* BGE, KorSIM
```python
bash eval.sh
```

* Openai(ada)
```python
bash openai_eval.sh
```
