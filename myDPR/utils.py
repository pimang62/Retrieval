from glob import glob
from typing import List
from torch import tensor as T


def get_passage_file(p_id_list: List[int], passages_dir) -> str:
    """passage id를 받아서 해당되는 파일 이름을 반환합니다."""
    target_file = None
    p_id_max = max(p_id_list)
    p_id_min = min(p_id_list)
    for f in glob(f"{passages_dir}/*.p"):  # passages_dir/*-*.p
        s, e = f.split("/")[-1].split(".")[0].split("-")
        s, e = int(s), int(e)
        if p_id_min >= s and p_id_max <= e:
            target_file = f
    return target_file
