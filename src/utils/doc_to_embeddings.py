from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

indexpath = "data/vectorstore/"

def text_to_embeddings(texts: str, embeddings):
    """
    Converts text to embeddings and creates a Faiss index.

    Parameters:
    - texts: Text or list of texts to convert to embeddings

    Returns:
    - faiss_index_path: Path to the saved Faiss index
    """
    # Initialize a text splitter for processing long texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    
    # Create documents by splitting the provided texts
    documents = text_splitter.create_documents([texts])

    # Create a Faiss index from the embeddings
    faiss_index = FAISS.from_documents(documents, embeddings)
    
    # Save the Faiss index locally
    faiss_index_path = indexpath + 'temp-index'
    faiss_index.save_local(faiss_index_path)
    
    return(faiss_index_path)  # Return the path to the saved Faiss vectorstore index
    
    






