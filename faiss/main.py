import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore

import gradio as gr

load_dotenv()

# config
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                          task_type="retrieval_document")
llm = ChatGoogleGenerativeAI(model="gemini-pro")

#creating faiss cache
faiss_cache = FAISS(
            embedding_function=embeddings,
            index=IndexFlatL2(768),
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
cache_size: int = 1
cache_max_size = 10

def isCacheFull():
    return cache_size==cache_max_size

def chat(q):
    global cache_size
    docs = faiss_cache.similarity_search_with_score(q)
    doc, score = docs[0] if docs else (0,1)
    print(f"\n Question: >> {q}")

    if score <= 0.075:
        print('Retrieved from Cache')
        result = doc.metadata['response']
        result = "Returned from Cache\n\n"+result
    else:
        result = (llm.invoke(q)).content
        print('Retrieved from LLM Call')

        if isCacheFull():
            faiss_cache.delete([faiss_cache.index_to_docstore_id[0]])
            cache_size-=1

        faiss_cache.add_documents([Document(page_content=q,
                                            metadata={"response": result}),])
        cache_size +=1
        result = "Returned from LLM call\n\n"+result
    return(result)


print(faiss_cache.index)

# Gradio

chatbot = gr.Interface(
    fn=chat,
    inputs=["text"],
    outputs=["text"],
)

demo = gr.Blocks()

with demo:
    gr.TabbedInterface(
      [chatbot])

demo.launch(share=True,
            debug=True)
