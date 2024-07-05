import pandas as pd
from marqodb import similarity_search_with_score
from chroma import similarity_search_chroma
from typing import (
    Dict,
    List,
    Tuple
)

from langchain.docstore.document import Document

questions = ["What is Projectile Motion?",
             "What are Newton’s three laws of motion?",
             "Explain Newton’s Laws of motion",
             "Explain Potential Energy",
             "what is centre of mass?",
             "Explain Kinematics and Dynamics of rotational motion about a fixed axis",
             "What is gravitational Constant",
             "Explain acceleration with respect to earths gravity",
             "say something about satellites on earth.",
             "explain relation between angular velocity and linear velocity",
             "Explain newton’s cooling law",
             "explain some mechanical properties of fluids",
             "what is thermodynamics",
             "Tell me about waves",
             "who found Matter is made up of atoms?",
             "Types of waves on surface of water?"]

indexes = ["index1", "index2", "index4", "index5", "index6", "index7"]

data = {
    'question': questions,
    'index1': [],
    'index2': [],
    'index4': [],
    'index5': [],
    'index6': [],
    'index7': [],
}

from fastapi import FastAPI

app = FastAPI()

@app.post("/query_marqo/")
def get_data(q: str) -> List[Tuple[Document, float]]:
    try:
     response = similarity_search_with_score(query=q,
                                        collection_name='index2')
     return response
    except Exception as e:
       print(e)
    return None

@app.post("/create_csv_marqo")
def create_csv() -> str:
    try:
        for index in indexes:
            for q in questions:
                response = similarity_search_with_score(query=q, collection_name=index, k=10)
                data[index].append(response)
            print(f"Index {index} completed")

        df = pd.DataFrame(data)
        file_name = 'mycsv.csv'
        df.to_csv(file_name, index=False)
        print(f"DataFrame saved to {file_name}")
        return "CSV created"
    except Exception as e:
        print(e)
        return "Error creating CSV"
    
@app.post("/query_chroma/")
def get_data(q: str) -> Dict:
    try:
     response = similarity_search_chroma(query=q,
                                        collection_name='index1')
     return response
    except Exception as e:
       print(f'>>>> Error searching Chromadb {e}')
    return None