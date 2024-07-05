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

load_dotenv()

client = chromadb.HttpClient(host=os.environ["CHROMA_HOST"], port=os.environ["CHROMA_PORT"])
embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
embedding_function = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")

# id generator for metadata
def generate_id():
    return str(uuid.uuid4().int)

def add_documents_chroma(documents=List[Document], collection_name: str="index1", fresh_collection: bool = False) -> List[str]:
    if fresh_collection:
        try:
            client.delete_collection(name="collection_name")
            print("Existing Index successfully deleted.")
        except:
            print("Index does not exist. Creating new index...")

    collections = client.get_or_create_collection(name=collection_name,
                                                    embedding_function=embedding_func)
    print(f"Index {collection_name} created.")

    docs: List[str] = []
    metadatas = []
    ids = []

    for d in documents:
        docs.append(d.page_content)
        metadatas.append(d.metadata)
        ids.append(generate_id())
    
    try:
        collections.add(
            documents=docs,
            metadatas=metadatas,
            ids=ids
        )
    except Exception as e:
        print(f'Error uploading documents to Chromdb {e}')
    return ids

def similarity_search_chroma(query: str, collection_name: str, k: int = 20) -> List[Tuple[Document, float]]:
    try:
        langchain_chroma = Chroma(client=client,
                                  collection_name="index1",
                                  embedding_function=embedding_function
                                  )
        response = langchain_chroma.similarity_search_with_score(query=query,
                                                      k=k)
        # collections = client.get_collection(collection_name,
        #                                     embedding_function=embedding_func)
        # response = collections.query(query_texts=[query])
        return response
    except Exception as e:
        print(f'>>>>> Error searching Chroma {e}')
    return {}