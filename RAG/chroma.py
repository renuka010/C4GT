import chromadb
import os
from dotenv import load_dotenv
import uuid
from typing import (
    Dict,
    List,
    Tuple
)
from langchain.docstore.document import Document
from chromadb.utils import embedding_functions
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_huggingface import HuggingFaceEmbeddings
from indexer import get_documents

load_dotenv()

client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"], port=os.environ["CHROMA_PORT"])
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
embedding_function = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# id generator for metadata
def generate_id():
    return str(uuid.uuid4().int)

def store_data_chroma(path:str, index:str, fresh_collection: bool) -> str:
    documents = get_documents(path)
    if fresh_collection:
        try:
            client.delete_collection(name=index)
            print("Existing Index successfully deleted.")
        except:
            print("Index does not exist. Creating new index...")
        collections = client.create_collection(name=index,
                                               embedding_function=embedding_func,
                                               metadata={"hnsw:space": "cosine"})
        print(f"Index {index} created.")
    else:
        collections = client.get_collection(name=index,
                                            embedding_function=embedding_func)
        print(f"Index {index} retrieved.")

    docs: List[str] = []
    doc_ids: List[str] = []
    metadatas = []

    for d in documents:
        docs.append(d.page_content)
        doc_ids.append(generate_id())
        metadatas.append({"subject": "physics"})

    print('Documents added')
    
    try:
        collections.add(
            documents=docs,
            metadatas = metadatas,
            ids=doc_ids
        )
        print("data added")
    except Exception as e:
        print(f'Error uploading documents to Chromdb {e}')
        return ""
    return "Data stored in ChromaDB"


def similarity_search_chroma(query: str, collection_name: str, k: int = 20) -> List[Tuple[Document, float]]:
    try:
        langchain_chroma = Chroma(client=client,
                                  collection_name=collection_name,
                                  embedding_function=embedding_function
                                  )
        response = langchain_chroma.similarity_search_with_score(query=query,
                                                                 k=k)
        return response
    except Exception as e:
        print(f'>>>>> Error searching Chroma {e}')
    return []