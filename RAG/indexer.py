from llama_index import SimpleDirectoryReader
import re
from typing import (
    List
)
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1024
CHUNK_OVERLAP = 200
file_name = "mydocs.txt"

def get_documents(path: str):
    documents = SimpleDirectoryReader(input_dir=path, recursive=True).load_data()
    document = ""
    for page in documents:
        document += (page.text+" ")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, 
                                                   chunk_overlap=CHUNK_OVERLAP,
                                                   separators=["."])
    splited_docs = []
    pattern = re.compile(r'[\x00-\x1F\x7F\u00A0]')
    with open(file_name, "w") as file:
        for chunk in text_splitter.split_text(document):
            chunk = re.sub(pattern, ' ', chunk)
            file.write(chunk+"\n\n")
            splited_docs.append(Document(page_content=chunk))
    return splited_docs