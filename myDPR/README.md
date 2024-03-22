## Run

1. Data Chunk 
* Included meta data extract
```
bash chunker.sh
```

2. Train
* By QA pair
```
python trainer.py
```

3. Create index
```
python index_runner.py
```

4. Retriever
```
python retriever.py -q "비밀번호를 찾고 싶어요." -k 3 > logs/retrieve.log
```