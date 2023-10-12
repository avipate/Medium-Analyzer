from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


if __name__ == "__main__":
    print("Hello")
    pdf_path = "pdf/vector_db.pdf"
    loader = PyPDFLoader(file_path=pdf_path)
    # Load the document
    documents = loader.load()

    # Split the characters
    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )
    docs = text_splitter.split_documents(documents=documents)

    # OpenAIEmbedding
    embeddings = OpenAIEmbeddings()

    """Faiss library helps us turn objects like pdf and text file to perform similarity search, 
     that will help us to feed oru LLM with more context."""
    # Take the embeddings and turn the documents which are all chunky into the vectors
    # This vectors can be later used to feed the LLM models.
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings=embeddings)

    # Chain to connect the vectors OpenAI model and prompts
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=new_vectorstore.as_retriever())

    res = qa.run("Give me the use case of vector database.")
    print(res)
