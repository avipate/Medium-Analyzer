# Importing required libraries
import os
from dotenv import load_dotenv, find_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain import OpenAI
import pinecone


pinecone.init(api_key="62e0f242-5aa4-4987-b98c-0a5db0543de8", environment="gcp-starter")
load_dotenv(find_dotenv())


if __name__ == "__main__":
    print("Hello VectorStore")
    loader = TextLoader("mediumblog.txt")
    document = loader.load()
    # To make a chunks of data which contains text from the document.
    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))
    # Embedding the vectors using OpenAI and then passing it into the Pinecone to form vectors
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(texts, embeddings, index_name="medium-blogs-eembedding-index")

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever()
    )
    query = "What is a vector DB? Give me a 15 word answer for a beginner."
    result = qa({"query": query})
    print(result)
    