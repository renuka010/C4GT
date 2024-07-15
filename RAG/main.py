from fastapi import FastAPI
from chroma import store_data_chroma, similarity_search_chroma
from langchain.docstore.document import Document
from typing import (
    Dict,
    List,
    Tuple
)

app = FastAPI()

@app.post("/store_chroma/")
def store_data(path: str, index:str, fresh:bool) -> str:
    return store_data_chroma(path,index,fresh)

@app.post("/search_chroma/")
def search_chroma(q: str) -> List[Tuple[Document, float]]:
    try:
        return similarity_search_chroma(query=q,collection_name="index3")
    except Exception as e:
        print(f'>>>>> Error {e}')
        return []