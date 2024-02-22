from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
from PyPDF2 import PdfReader
from datetime import datetime
from typing import Optional
from io import BytesIO
import chainlit as cl
import sys

# environment for the app
# conda activate llama2Apps
# command to run the app
# chainlit run src/apps/localLLM_withRAG-Complete.py --port 8001 -w

prompt_template = """You are an helpful AI assistant and your name is SAHAYAK. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in concise and step by step manner. 
If you don't know the answer to a question, please don't share false information.
            
Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""
modelpath = "../models/llama-2-7b-chat.Q2_K.gguf"

# model used for converting text/queries to numerical embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

# path to store embeddings at vectorstore
indexpath = "data/vectorstore/"

# number of neural network layers to be transferred to be GPU for computation 
n_gpu_layers = 10
n_batch = 256

config = {'max_new_tokens': 512, 'context_length': 4096,         
            'gpu_layers': n_gpu_layers,'batch_size': n_batch,   
            'temperature': 0.1
         }
# Initialize embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

@cl.cache
def load_llama2_llm(modelpath):
    """
    Loads a Llama2 language model from the specified model path.

    Parameters:
    - modelpath: Path to the Llama2 language model
    Returns:
    - llm: Llama2 language model instance
    """
    # Create a Llama2 language model using CTransformers
    llm = CTransformers(
        model=modelpath,       # Path to the Llama2 model
        model_type="llama",    # Specify the model type as "llama"
        config=config          # Configuration for the model, like context length, max new tokens, temperature etc.
    )
    return(llm) 

# Loading the local model into LLM
llm = load_llama2_llm(modelpath)
    
@cl.on_chat_start
async def factory():
    # loads the data by the user
    files = None

    ### wait for the user to upload a data file
    while files == None:
        files = await cl.AskFileMessage(
            content="""Your personal AI asistant, SAHAYAK is ready to slog!
                     To get started:
                     
1. Upload a pdf file                     
2. Ask any questions about the file!""",
                    accept={"application/pdf": [".pdf"]
                            },
                     max_size_mb=10
        ).send()

    # Let the user know that the system is ready
    await cl.Message(
        content=f"""Document - `"{files[0].name}"` is uploaded and being processed!"""
    ).send()

    
    ### Reads and convert pdf data to text 
    file = files[0]
    # Convert the content of the PDF file to a BytesIO stream
    text_stream = BytesIO(file.content)
    # Create a PdfReader object from the stream to extract text 
    pdf = PdfReader(text_stream)  
    pdf_text = ""
    # Iterate through each page in the PDF and extract text
    for page in pdf.pages:
        pdf_text += page.extract_text()  # Concatenate the text from each page

    
    ### Create embeddings for the uploaded documents and store in vector store
    # Initialize a text splitter for processing long texts
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, 
                                                   chunk_overlap=10)
    # Create documents by splitting the provided texts
    documents = text_splitter.create_documents([pdf_text])
    # Create a Faiss index from the embeddings
    faiss_index = FAISS.from_documents(documents, embeddings)
   
    # Save the Faiss index locally
    faiss_index_path = indexpath + 'temp-index'
    faiss_index.save_local(faiss_index_path)
    # Load Faiss vectorstore with embeddings created and saved earlier
    db = FAISS.load_local(faiss_index_path, embeddings)
    
    prompt = PromptTemplate(template=prompt_template,
                       input_variables=['context', 'question'])
    
    # Creating a retrieval QA chain using the provided llm, chain type, retriever, and prompt
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Replace with the actual chain type
        retriever=db.as_retriever(search_kwargs={'k': 1}),  # Assuming vectorstore is used as a retriever
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    
    msg = cl.Message(content="The bot is getting initialized, please wait!!!")
    await msg.send()
    msg.content = "Your personal AI Assistant, SAHAYAK is ready. Ask questions on the documents uploaded?"
    await msg.update()
    cl.user_session.set("chain", chain)


@cl.on_message
async def main(message):
    start_time = datetime.now()
    chain = cl.user_session.get("chain")
    response = await chain.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("total time taken was:", time_taken)
    
    await cl.Message(content=response["result"]).send()