from langchain_text_splitters import Language,RecursiveCharacterTextSplitter
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_community.document_loaders.blob_loaders import Blob, BlobLoader
from langchain_core.documents import Document
from io import TextIOWrapper
from inserter import ChromaCollection
import sys
import os 
from zipfile import ZipFile


_extension_map={
".ts":"ts",
".js":"js",
".java":"java",
".py":"python",
".c":"c",
".cpp":"cpp",
".h":"c",
".cs":"csharp",
".html":"html",
".go":"go",
".php":"php",
".kt":"kotlin",
".hs":"haskell",
".tex":"latex",
".lua":"lua",
".rb":"ruby",
".rust":"rust",
".md":"markdown"
}

_excluded_extensions={".png",".svg","/"}

def parse_file(filepath:str,content:str,parser_threshold=3):
    parser=LanguageParser(parser_threshold=parser_threshold)
    blob=Blob(path=filepath,encoding="utf-8", data=content)
    doc=parser.lazy_parse(blob=blob)
    return list(doc)

    

def load_from_zipfile(zippath:str):
    zip_documents=[]
    with ZipFile(zippath,'r') as z:
        for filename in z.namelist():
            if filename.startswith(".") or os.path.splitext(filename)[1] in _excluded_extensions:
                continue
            with z.open(filename,"r") as f:
                print(filename)
                text=TextIOWrapper(f,encoding="utf-8").read()
            doc=parse_file(filename,text)
            zip_documents.append(doc)
    return zip_documents


if __name__=="__main__":
    chroma=ChromaCollection(sys.argv[1])
    docs=load_from_zipfile(sys.argv[2])
    for doc in docs:
        chroma.insert_docs(doc)
    retreived,_=chroma.retrieve_k_similar_docs("what is the authentication logic",k=10)
    #res=chroma.rag(query="where is the authentication logic ?")
    print("=xd"*60)
    for ret in retreived:
        print(ret.metadata)