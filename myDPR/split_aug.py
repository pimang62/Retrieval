"""
baemin_passage :
DatasetDict({
    Dataset({
        features: ['doc_id', 'text'],
        num_rows: 268893
    })
})

baemin_queries:
DatasetDict({
    Dataset({
        features: ['qid', 'query', 'answers'],
        num_rows: 2076
    })
})

"""

import json
from glob import glob

# baemin_passage = {
#     "doc_id": [],
#     "text": []
# }

# baemin_queries = {
#     "qid": [],
#     "query": [],
#     "answers" : []
# }

def main():
    idx = 0
    for i in range(137):  # 0~136
        for path in glob(f"augmented/{i}-*-0.json"):
            with open(path, 'r') as f:
                data = json.load(f)
            
                doc_id = data["document_id"]
                text = data["content"]

                baemin_passage = {
                    "doc_id": doc_id,
                    "text": text
                }
                with open("dataset/baemin_passage.json", 'w') as _f:
                    json.dump(baemin_passage, _f, indent=2, ensure_ascii = False)
                return  # 3/15 이상..

        # query = [data["question"]] + data["generated_questions"]  # List[str]
        # qid = [i for i in range(idx+len(query))]

if __name__=='__main':
    main()