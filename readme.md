# Retrieval(Vector Search)
---

문서 검색 QA task

## Setup
---

### Install
`git clone `
`conda create -n exper python==3.10`
`conda activate exper`
`pip install -r requirements.txt`
* Reload the session!

## Data
---

private

## Run
---

### Chunking & Vector mapping
* BGE, KorSIM, Openai(ada)
  * Openai has only "txt" file type
`bash create_index.sh`

### Run demo
`cd src`
`bash run_demo.sh`

### Show demo
`cat demo.log`

### Evaluation
* BGE, KorSIM
`bash eval.sh`

* Openai(ada)
`bash openai_eval.sh`