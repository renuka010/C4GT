import argparse
import re
from typing import (
    List
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llama_index import SimpleDirectoryReader
from marqodb import add_documents
from chroma import add_documents_chroma, similarity_search_chroma
 
def document_loader(input_dir: str) -> List[Document]:
    return SimpleDirectoryReader(
        input_dir=input_dir, recursive=True).load_data() # show_progress=True 
 
def split_documents(documents: List[Document], chunk_size: int = 4000, chunk_overlap = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["."])
    # splited_docs = text_splitter.split_documents(documents)
    splited_docs = []
    pattern = re.compile(r'[\x00-\x1F\x7F\u00A0]')
    for document in documents:
        for chunk in text_splitter.split_text(document.text):
            chunk = re.sub(pattern, '', chunk)
            splited_docs.append(Document(page_content=chunk, metadata={
                "subject":"physics"
            }))
    return splited_docs

def transform_documents():
    pass

def load_documents(folder_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    documents = document_loader(folder_path)
    splitted_documents = split_documents(documents, chunk_size, chunk_overlap)
    return splitted_documents

def indexer_marqo():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path',
                        type=str,
                        required=False,
                        help='Path to the folder',
                        default="input_data/XI_books/XI_Physics_Part2"
                        )
    # parser.add_argument('--index_name',
    #                     type=str,
    #                     required=True,
    #                     help='Collection name for index',
    #                     default="index1"
    #                     )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )

    args = parser.parse_args()

    FOLDER_PATH = args.folder_path
    COLLECTION_NAMES = ["passcurr_split"]
    FRESH_INDEX = args.fresh_index
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200

    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    for COLLECTION_NAME in COLLECTION_NAMES:
        print("Total documents :: =>", len(documents))
        
        print("Adding documents...")
        results = add_documents(documents=documents, collection_name=COLLECTION_NAME, fresh_collection=FRESH_INDEX)
        print("results =======>", results)
        
        print(f"============ INDEX {COLLECTION_NAME} DONE =============")


def indexer_chroma():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path',
                        type=str,
                        required=False,
                        help='Path to the folder',
                        default="input_data/XI_books/XI_Physics_Part2"
                        )
    # parser.add_argument('--index_name',
    #                     type=str,
    #                     required=True,
    #                     help='Collection name for index',
    #                     default="index1"
    #                     )
    parser.add_argument('--fresh_index',
                        action='store_true',
                        help='Is the indexing fresh'
                        )

    args = parser.parse_args()

    FOLDER_PATH = args.folder_path
    COLLECTION_NAMES = ["index1"]
    FRESH_INDEX = args.fresh_index
    CHUNK_SIZE = 1024
    CHUNK_OVERLAP = 200

    documents = load_documents(FOLDER_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    for COLLECTION_NAME in COLLECTION_NAMES:
        print("Total documents :: =>", len(documents))
        
        print("Adding documents...")
        results = add_documents_chroma(documents=documents, collection_name=COLLECTION_NAME, fresh_collection=FRESH_INDEX)
        print("results =======>", results)
        
        print(f"============ INDEX {COLLECTION_NAME} DONE =============")
        q= "What is projectile motion?"
        print(f'Query: {q}')
        similarity_search_chroma(query=q,collection_name='index10',k = 10)
        

        
if __name__ == "__main__":
    indexer_marqo()
    
# For Fresh collection
# python3 indexer.py --folder_path=input_data/XI_books --index_name=index1 --fresh_index