import os
from typing import List, Tuple
from kobert_tokenizer import KoBERTTokenizer


class BaeminDataset:
    def __init__(self, baemin_path: str, 
                 baemin_title_passage_map_path: "baemin_title_passage_map.p"):
        self.baemin_path = baemin_path
        self.data_tuples = []
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")
        self.pad_token_id = self.tokenizer.get_vocat()["[PAD]"]
        self.load()
    
    @property  # BaeminDataset.dataset -> tokenized_tuples
    def dataset(self) -> List[Tuple]:
        return self.tokenized_tuples

    def load(self):
        self.baemin_processed_path = (
            f"{self.baemin_path.split("./json")[0]}_processed.p"
        )
        if os.path.exists(self.baemin_path)
