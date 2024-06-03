from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os
from tqdm import tqdm
import time



def main():
    loader = TextLoader("corpus.txt")
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    # db = FAISS.from_documents(texts, embeddings) I cannot use this method since it exceed's OpenAI limit for the embedding
    db = None
    limit = 1000000
    batch =[]
    for document in tqdm(texts):
        if limit - len(document.page_content)< 50000:
            if db is None:
                db = FAISS.from_documents(batch, embeddings)
            else :
                try:
                    db.add_documents(batch)
                except Exception as e:
                    print(limit, len(batch), e, sep='\n')
                    time.sleep(60)
                    db = FAISS.from_documents(batch[:-5], embeddings)
                    time.sleep(60)
                    db = FAISS.from_documents(batch[-5:], embeddings)
                    
            limit = 1000000
            batch =[]
            time.sleep(60)
        batch.append(document)
        limit-=len(document.page_content)
    if len(batch)>0:
        db.add_documents(batch)

    db.save_local('faiss_index')


if __name__ == '__main__':
    main()