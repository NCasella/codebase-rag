import argparse
import sys
import os
from src.inserter import ChromaCollection   
from src.text_splitter import load_from_zipfile, load_from_github_link

if __name__ == "__main__":

    
    arg_parser=argparse.ArgumentParser(prog="codebase RAG")
    group=arg_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-z","--zip")
    group.add_argument("-g","--github_link")
    arg_parser.add_argument("-p","--prompt",required=True)
    arg_parser.add_argument("-c","--collection-name",default="codeRAG")

    
    args=arg_parser.parse_args()
    
    zip_path=args.zip
    github_link=args.github_link
    collection_name=args.collection_name
    user_prompt=args.prompt
    
    if zip_path:
        if not os.path.exists(zip_path):
            print(f"Error: No se encontró el archivo {zip_path}")
            sys.exit(1)
        docs = load_from_zipfile(zip_path)
    else:
        try:
            docs=load_from_github_link(github_link)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Indexando codebase: {zip_path}")
    print(f"Colección: {collection_name}")
    print(f"{'='*60}\n")

    chroma = ChromaCollection(collection_name)


    # Cargar documentos desde ZIP

    if len(docs) == 0:
        print("\n✗ No se encontraron documentos para indexar")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Total de fragmentos a insertar: {len(docs)}")
    print(f"{'='*60}\n")

    print("Insertando en ChromaDB...")
    chroma.insert_docs(docs)

    
    print(f"{'='*60}\n")

    res = chroma.rag(query=user_prompt)
    print("Respuesta:")
    print(res)
    print(f"\n{'='*60}")