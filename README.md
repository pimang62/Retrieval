# Retrieval

문서 검색 QA task

## Setup

### Install
```python
git clone https://github.com/pimang62/Retrieval.git
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
bash script/create_index.sh
```

### Run demo
```python
bash script/run_demo.sh
```

### Show demo
```python
cat script/demo.log
```
