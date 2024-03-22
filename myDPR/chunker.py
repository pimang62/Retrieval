import os
import logging
import re
import json
import pickle
from typing import Tuple, List
from kobert_tokenizer import KoBERTTokenizer
from glob import glob
from copy import deepcopy
from tqdm import tqdm
from collections import defaultdict

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


def post_processing(text:str) -> str:
    return re.sub(r'\.', '', text)

class DataChunk:
    """
    trainer용 input data 형태를 따르는 chunk 클래스입니다.
    >>> [{"passages": {}, "queries": [{}]}] 
    
    Option
    1. chunk_size: 100, 200, 50 ...
    2. chunk_overlap: 0, 5, 10, ...
    3. passages_type: "" or "content" or "answer"
    """
    def __init__(self, chunk_size: int=100, chunk_overlap: int=0, passages_type=""):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.passages_type = passages_type
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    def chunk(self, title_text: Tuple[str, str]) -> Tuple[List[str], List[int]]:
        title, text = title_text
        """chunk_size, chunk_overlap 유효"""
        if self.passages_type == "":
            chunk_list = []
            orig_text = []

            encoded_title = self.tokenizer.encode(title)
            encoded_text = self.tokenizer.encode(text)

            for start_idx in range(0, len(encoded_text), self.chunk_size-self.chunk_overlap):
                end_idx = min(len(encoded_text), start_idx + self.chunk_size)
                chunk = encoded_title + encoded_text[start_idx:end_idx]
                orig_text.append(self.tokenizer.decode(chunk))
                chunk_list.append(chunk)
        
        else:   # "content" or "answer"
            chunk_list = []
            orig_text = []

            encoded_title = self.tokenizer.encode(title)
            encoded_text = self.tokenizer.encode(text)
            
            chunk = encoded_title + encoded_text

            orig_text.append(self.tokenizer.decode(chunk))
            chunk_list.append(chunk)

        return orig_text, chunk_list



def save_orig_passage(
    data_path,
    passages_dir, 
    chunk_size, 
    chunk_overlap, 
    passages_type
):
    """store original passages with unique id"""
    os.makedirs(passages_dir, exist_ok=True)
    chunker = DataChunk(chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap, 
                        passages_type=passages_type)
    
    if (passages_type == "") or (passages_type == "content"):
        idx = 0
        for sample in json.load(open(data_path, 'r')):
            title_text = (sample["passages"]["title"], sample["passages"]["text"])
            ret, _ = chunker.chunk(deepcopy(title_text))
            to_save = {idx + i: ret[i] for i in range(len(ret))}
            with open(f"{passages_dir}/{idx}-{idx+len(ret)-1}.p", "wb") as f:
                pickle.dump(to_save, f)
            idx += len(ret)
    else:  # passages_type == "answer"
        idx = 0
        for sample in json.load(open(data_path, 'r')):
            title = sample["passages"]["title"]
            answers = sample["queries"]
            ret = []  # 한 문서당 qa pairs
            for answer in answers:
                title_text = (title, answer["answer"])
                arc, _ = chunker.chunk(deepcopy(title_text))
                ret.append(arc[0])  # [[]] -> []
            to_save = {idx + i: ret[i] for i in range(len(ret))}
            with open(f"{passages_dir}/{idx}-{idx+len(ret)-1}.p", "wb") as f:
                pickle.dump(to_save, f)
            idx += len(ret)
    return


def save_title_index_map(
    title_index_map_path, 
    passages_dir
):
    """korquad와 klue 데이터 전처리를 위해 title과 passage id를 맵핑합니다.
    title_index_map : dict[str, list] 형태로, 특정 title에 해당하는 passage id를 저장합니다.
    """
    logging.getLogger()

    files = glob(f"{passages_dir}/*")
    title_id_map = defaultdict(list)
    for file in tqdm(files):
        with open(file, "rb") as _f:
            id_passage_map = pickle.load(_f)
        for id, passage in id_passage_map.items():
            title = passage.split("[SEP]")[0].split("[CLS]")[1].strip()
            title_id_map[title].append(id)
        logger.info(f"processed {len(id_passage_map)} passages from {_f}...")
    with open(f"{title_index_map_path}", "wb") as f:
        pickle.dump(title_id_map, f)
    logger.info(f"Finished saving title_index_mapping at {title_index_map_path}!")


class ArticleChunk:
    """
    index_runner용 Article을 자르는 chunk 클래스입니다.
    >>> data_path가 들어옵니다.
    """
    def __init__(self, chunk_size: int=None, chunk_overlap: int=None, passages_type=""):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.passages_type = passages_type
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    def chunk(self, data_path):
        """input file format은 baemin(_nohead).txt 형태를 따릅니다."""
        with open(data_path, "rt", encoding="utf8") as f:
            input_txt = f.read().strip()
        input_txt = input_txt.split(
            "\n\n"
        )  # 문단 단위로 split하여 각 article의 제목과 본문을 parsing합니다.
        
        chunk_list = []
        orig_text = []
        for art in input_txt:
            art = art.strip()
            if not art:
                logger.debug(f"article is empty, passing")
                continue

            encoded_txt = self.tokenizer.encode(art)
            if len(encoded_txt) < 5:  # 본문 길이가 subword 5개 미만인 경우 패스
                logger.debug(f"article: {art} has <5 subwords in its article, passing")
                continue

            # article마다 chunk_size 길이의 chunk를 만들어 list에 append. (각 chunk에는 title을 prepend합니다.)
            # ref : DPR paper
            if self.chunk_size is not None:
                for start_idx in range(0, len(encoded_txt), self.chunk_size-self.chunk_overlap):
                    end_idx = min(len(encoded_txt), start_idx + self.chunk_size)
                    chunk = encoded_txt[start_idx:end_idx]
                    orig_text.append(self.tokenizer.decode(chunk))
                    chunk_list.append(chunk)
            else:
                orig_text.append(art)
                chunk_list.append(encoded_txt)

        return orig_text, chunk_list


if __name__=='__main__':
    """"
    python chunker.py --data_path "dataset/baemin.json" \
                      --passages_dir "baemin_passages_" \
                      --title_index_map_path "baemin_title_passage_map_.p"
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="dataset/baemin.json", required=True)
    parser.add_argument("--passages_dir", default="baemin_passages_", required=True)
    parser.add_argument("--title_index_map_path", default="baemin_title_passage_map_.p", required=True)
    parser.add_argument("--passages_type", default="", required=True)
    parser.add_argument("--chunk_size", default=100)
    parser.add_argument("--chunk_overlap", default=0)

    args = parser.parse_args()

    data_path = args.data_path
    passages_dir = args.passages_dir
    title_index_map_path = args.title_index_map_path
    passages_type = args.passages_type
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    # data_path = "dataset/baemin.json"
    # passages_dir = "baemin_passages_"
    # title_index_map_path = "baemin_title_passage_map_.p"
    # chunk_size = 100
    # chunk_overlap = 50
    # passages_type = ""
    
    # save_orig_passage(
    #     data_path=data_path,
    #     passages_dir=passages_dir, 
    #     chunk_size=chunk_size, 
    #     chunk_overlap=chunk_overlap, 
    #     passages_type=passages_type
    # )

    # save_title_index_map(
    #     title_index_map_path=title_index_map_path, 
    #     passages_dir=passages_dir
    # )

    # # content
    # data_path = "dataset/baemin.json"
    # passages_dir = "baemin_passages_content"
    # title_index_map_path = "baemin_title_passage_map_content.p"
    # passages_type = "content"

    # save_orig_passage(
    #     data_path=data_path,
    #     passages_dir=passages_dir, 
    #     chunk_size=100,  # any value
    #     chunk_overlap=0, 
    #     passages_type=passages_type
    # )

    # save_title_index_map(
    #     title_index_map_path=title_index_map_path, 
    #     passages_dir=passages_dir
    # )

    # # answer
    # data_path = "dataset/baemin.json"
    # passages_dir = "baemin_passages_answer"
    # title_index_map_path = "baemin_title_passage_map_answer.p"
    # passages_type = "answer"

    save_orig_passage(
        data_path=data_path,
        passages_dir=passages_dir, 
        chunk_size=100,  # any value
        chunk_overlap=0, 
        passages_type=passages_type
    )

    save_title_index_map(
        title_index_map_path=title_index_map_path, 
        passages_dir=passages_dir
    )
