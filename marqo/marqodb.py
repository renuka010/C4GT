import json
import os
from dotenv import load_dotenv
from typing import (
    Dict,
    List,
    Tuple
)

import marqo
from langchain.docstore.document import Document
from langchain.vectorstores.marqo import Marqo


load_dotenv()

SPLIT_LENGTH: int = 1
SPLIT_OVERLAP: int = 0
BATCH_SIZE: int = 50
TENSOR_FIELDS: str = ["text"]
client_url = os.environ["VECTOR_STORE_ENDPOINT"]
client: marqo.Client = marqo.Client(url=client_url)
embedding_models: Dict = {    
    "sent_split": "sentence-transformers/all-mpnet-base-v2",
    "char_split": "sentence-transformers/all-mpnet-base-v2",
    "passage_split": "sentence-transformers/all-mpnet-base-v2",
    "passcurr_split" : "sentence-transformers/all-mpnet-base-v2",

    "index1" : "sentence-transformers/all-MiniLM-L6-v2", #384
    "index2" : "sentence-transformers/all-mpnet-base-v2",  #768
    "index4" : "flax-sentence-embeddings/all_datasets_v3_MiniLM-L6", #384
    "index5" : "flax-sentence-embeddings/all_datasets_v4_MiniLM-L6", #384
    "index6" : "flax-sentence-embeddings/all_datasets_v3_mpnet-base", #768
    "index7" : "flax-sentence-embeddings/all_datasets_v4_mpnet-base", #768
    "index8" : "flax-sentence-embeddings/all_datasets_v4_mpnet-base",
    "index9" : "sentence-transformers/all-mpnet-base-v2",
    "index10" : "sentence-transformers/all-mpnet-base-v2",
}

def chunk_list(document: List, batch_size: int) -> List[List]:
    return [document[i: i + batch_size] for i in range(0, len(document), batch_size)]

def add_documents(documents=List[Document], collection_name: str="index1", fresh_collection: bool = False) -> List[str]:
    if fresh_collection:
        try:
            client.index(collection_name).delete()
            print("Existing Index successfully deleted.")
        except:
            print("Index does not exist. Creating new index...")

        index_settings = {
            "treatUrlsAndPointersAsImages": False,
            "model": embedding_models.get(collection_name),
            "normalizeEmbeddings": True,
            "textPreprocessing": {
                "splitLength": SPLIT_LENGTH,
                "splitOverlap": SPLIT_OVERLAP,
                "splitMethod": "passage"
            }
        }
        client.create_index(
            index_name=collection_name, settings_dict=index_settings)
        print(f"Index {collection_name} created.")

    docs: List[Dict[str, str]] = []
    ids = []
    for d in documents:
        doc = {
            "text": (d.page_content+"\n"),
            "metadata": json.dumps(d.metadata) if d.metadata else json.dumps({}),
        }
        docs.append(doc)
    chunks = list(chunk_list(docs, BATCH_SIZE))
    for chunk in chunks:
        response = client.index(collection_name).add_documents(
            documents=chunk, client_batch_size=BATCH_SIZE, tensor_fields=TENSOR_FIELDS)
        if response[0]["errors"]:
            err_msg = (
                f"Error in upload for documents in index range"
                f"check Marqo logs."
            )
            raise RuntimeError(f'Error : --- {response}')
        ids += [item["_id"] for item in response[0]["items"]]

    return ids

def similarity_search_with_score(query: str, collection_name: str, k: int = 20) -> List[Tuple[Document, float]]:
    try:
        docsearch = Marqo(client, index_name=collection_name)
        documents = docsearch.similarity_search_with_score(query, k)
        
        # reranker_model = CrossEncoder("jinaai/jina-reranker-v1-turbo-en", trust_remote_code=True)
        # #query_and_docs = [(query, r.page_content) for r in documents]
        # #scores = reranker_model.predict(query_and_docs)
        # #return sorted(list(zip(documents, scores)), key=lambda x: x[1], reverse=True)

        # docs = [d[0].page_content for d in documents]
        # results = reranker_model.rank(query, docs, return_documents=True)
    except Exception as e:
        print(f'>>>>> {e}')
    return documents
