'''
data["content"] 전부 붙여서 chunk 후 indexing
'''

import json
from glob import glob
from tqdm import tqdm
from kobert_tokenizer import KoBERTTokenizer
from metadata_extract import create_metadata
from collections import defaultdict
import os
import pickle
import logging

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/log.log",
    level=logging.DEBUG,
    format="[%(asctime)s | %(funcName)s @ %(pathname)s] %(message)s",
)
logger = logging.getLogger()


class BaeminChunk:
    def __init__(self, chunk_size=100):
        self.chunk_size = chunk_size
        self.tokenizer = KoBERTTokenizer.from_pretrained("skt/kobert-base-v1")

    def chunk(self, input_file):
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        """내부 정보 load"""
        doc_id = data["document_id"]
        query = [data["question"]] + data["generated_questions"]
        
        content = data["content"]
        title = create_metadata(content)

        chunk_list = []
        orig_text = []
        encoded_title = self.tokenizer.encode(title)
        encoded_txt = self.tokenizer.encode(content)

        for start_idx in range(0, len(encoded_txt), self.chunk_size):
            end_idx = min(len(encoded_txt), start_idx + self.chunk_size)
            chunk = encoded_title + encoded_txt[start_idx:end_idx]
            orig_text.append(self.tokenizer.decode(chunk))
            chunk_list.append(chunk)
        return orig_text, chunk_list

def save_orig_passage(
    passage_path="baemin_passages", chunk_size=100
):
    """store original passages with unique id"""
    os.makedirs(passage_path, exist_ok=True)
    app = BaeminChunk(chunk_size=chunk_size)
    idx = 0
    for num in range(137):  # 각 content가 담긴 파일을 모두 돌며
        for path in (glob(f"../augmented/{num}-*-0.json")):
            ret, _ = app.chunk(path)
            to_save = {idx + i: ret[i] for i in range(len(ret))}
            # print(to_save[0])  # [CLS] 정지운 선생 묘[SEP][CLS] 정지운 선생 묘는 경기도 고양시 일산동구 중산동에 있는 조선시대의 무덤이다. 1986년 6월 16일 고양시의 향토문화재 제11호로 지정되었다. 개요. 묘는 일산동 중산(中山) 마을 고봉산 남쪽 기[UNK]에 위치하고 있으며 배([UNK]) 정부인([UNK]人) 충주 안씨([UNK]州 安[UNK])의 묘와 쌍
            with open(f"../{passage_path}/{idx}-{idx+len(ret)-1}.p", "wb") as f:
                pickle.dump(to_save, f)
            idx += len(ret)


def save_title_index_map(
    index_path="baemin_title_passage_map.p", source_passage_path="baemin_passages"
):
    """korquad와 klue 데이터 전처리를 위해 title과 passage id를 맵핑합니다.
    title_index_map : dict[str, list] 형태로, 특정 title에 해당하는 passage id를 저장합니다.
    """
    logging.getLogger()

    files = glob(f"../{source_passage_path}/*")
    title_id_map = defaultdict(list)
    for f in tqdm(files):
        with open(f, "rb") as _f:
            id_passage_map = pickle.load(_f)
        for id, passage in id_passage_map.items():
            title = passage.split("[SEP]")[0].split("[CLS]")[1].strip()
            title_id_map[title].append(id)
        logger.info(f"processed {len(id_passage_map)} passages from {f}...")
    with open(f"../{index_path}", "wb") as f:
        pickle.dump(title_id_map, f)
    logger.info(f"Finished saving title_index_mapping at {index_path}!")

if __name__ == '__main__':
    save_orig_passage()
    save_title_index_map()
    # with open('/home/pimang62/projects/ir/KorDPR/baemin_title_passage_map.p', 'rb') as f:
    #     data = pickle.load(f)
    # print(next(iter(data)))
