import os
import chromadb
from openai import OpenAI
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv, find_dotenv
from langchain_core.documents import Document
from uuid import uuid4

_=load_dotenv(find_dotenv())

_embedding_function = SentenceTransformerEmbeddingFunction()  # Es una extension de BERT
_chroma_client = chromadb.Client()


class ChromaCollection():
    def __init__(self,collection_name) -> None:
        self._chroma_collection=_initialize_collection(collection_name=collection_name)
        self.openai_client=OpenAI(api_key=os.environ["OPENAI_API_KEY"])#TODO hacerlo independiente del modelo.

    def retrieve_k_similar_docs(self,queries,k=5):
        results = self._chroma_collection.query(query_texts=queries, n_results=k, include=['documents', 'embeddings'])
        retrieved_documents = results['documents'][0]
        return retrieved_documents, results
    
    def insert_docs(self,docs:list[Document]):
        docs_list=list(docs or [])
        if len(docs_list)==0:
            print("docs vacio")
            return
        self._chroma_collection.add(ids=[str(uuid4()) for _ in docs],
                                    documents=[doc.page_content for doc in docs],metadatas=[doc.metadata for doc in docs])
    
    def rag(self, query, model="gpt-4.1-nano"):
        documents,_=self.retrieve_k_similar_docs(query,k=10)
        information="\n".join(documents)
        messages=[
            {"role":"system",
            "content":"You are an expert agent in coding. Users will ask questions involving a code base. You will be shown the user's question and the relevant code involving the question"
            "Use the information given to answer the user's question"},
            {"role":"user",
             "content":f"Question:{query}. \n Information: {information }"}
        ]
        response = self.openai_client.chat.completions.create( model=model,messages=messages,)
        content = response.choices[0].message.content
        return content

def _initialize_collection(collection_name:str) -> chromadb.Collection:
    existing_collections = _chroma_client.list_collections()
    existing_collection_names = [collection.name for collection in existing_collections]
    if collection_name not in existing_collection_names:
        chroma_collection = _chroma_client.create_collection(
            name=collection_name,
            embedding_function=_embedding_function)
    else:
        chroma_collection = _chroma_client.get_collection(name=collection_name)
    return chroma_collection
