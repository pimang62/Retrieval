import yaml
import streamlit as st
import sys
sys.path.append("/home/pimang62/projects/ir/Retrieval/")
from app.src.demo.vectorstore import VectorStore
from app.src.demo.visualise import visualise

yaml_path = "/home/pimang62/projects/ir/Retrieval/app/src/demo/retrieval_config.yaml"

def main():
    """streamlit demo 실행"""
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    st.title("Retrieval Test")

    user_input = st.text_input("Enter text")

    # database selection
    database_options = {v: k for k, v in config["database"].items()}
    database = st.selectbox(
        "Select the Database to search from",
        tuple(config["database"].values())
        # ("openai text-embedding-ada", "bge", "korsim-bert", "skt-kobert", 
        #  "KorDPR (skt-kobert) with squad", "KorDPR (skt-kobert) with squad + faq data (ours)")
    )

    # embedding model
    embedding_options = {v: k for k, v in config["embedding"].items()}
    model_option = st.selectbox(
        "Select the Embedding Model",
        tuple(config["embedding"].values())
        # ("openai text-embedding-ada", "bge", "korsim-bert", "skt-kobert", 
        #  "KorDPR (skt-kobert) with squad", "KorDPR (skt-kobert) with squad + faq data (ours)")
    )


    # chunk size
    chunk_size = st.selectbox(
        "Select the Chunk Size",
        tuple(config['chunk_size'])
    )

    chunk_overlap = st.selectbox(
        "Select the Chunk overlap",
        tuple(config['chunk_overlap'])
    )

    top_k = st.selectbox(
        "Select the number of top k",
        tuple(config['top_k'])
    )

    distance_metric = st.selectbox(
        "Select the distance metric",
        tuple(config['distance_metric'])
    )


    if st.button("Get Related Documents"):
        vectorstore = VectorStore(database_name=database_options[database], 
                                  embedding_model_name_or_path=embedding_options[model_option],
                                  chunk_size=chunk_size, 
                                  chunk_overlap=chunk_overlap)

        retrieve_documents = vectorstore.retrieve(query=user_input, top_k=top_k)
        result_str = "\n\n".join([f"## 문서{i+1}\n {doc}" for i, doc in enumerate(retrieve_documents)])
        st.text(f"""Result:\n{result_str}""")

        with st.spinner('Creating visulisation...'):
            fig = visualise(query=user_input, 
                            top_k=top_k,
                            embedding_type=embedding_options[model_option], 
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            database=database_options[database])
        st.pyplot(fig)

if __name__ == "__main__":
    main()
