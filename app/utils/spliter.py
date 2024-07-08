import re

def filter_documents_only(database):
    """
    Q-A 쌍 데이터 => Q / A 데이터로만 필터링
    """
    fpath=f"../docx/{database}.txt"

    with open(fpath, "r") as f:
        text = f.read()
    
    head_passages = [t.split("\n")[0] for t in text.split("\n\n")]
    nohead_passages = ["\n".join(t.split("\n")[1:]) for t in text.split("\n\n")]
    
    with open(re.sub(".txt", "_head.txt", fpath), "w") as f:
        f.write("\n\n".join(head_passages))  # Question data

    with open(re.sub(".txt", "_nohead.txt", fpath), "w") as f:
        f.write("\n\n".join(nohead_passages))  # Answer data

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=str, default="baemin")
    
    args = parser.parse_args()
    database = args.database

    filter_documents_only(database)