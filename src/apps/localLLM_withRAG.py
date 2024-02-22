from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from datetime import datetime
from typing import Optional
import chainlit as cl
import sys

# load user defined utils
sys.path.append('src/utils/')
from conversation_utils import create_prompt, get_response_from_qa_chain, answering_bot
from load_data import pdf_to_text
from doc_to_embeddings import text_to_embeddings
from model_utils import load_llama2_llm

# environment for the app
# conda activate llama2Apps
# command to run the app
# chainlit run src/apps/localLLM_withRAG.py --port 8001 -w


modelpath = "../models/llama-2-7b-chat.Q2_K.gguf"
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
# Initialize embeddings using HuggingFace model
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

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
    # reads and convert pdf data to text 
    texts = pdf_to_text(file)
        
    # create embeddings for the uploaded document
    db_faiss_path = text_to_embeddings(texts, embeddings)
        
    # create llm chain for RAG usecase
    chain = answering_bot(llm, db_faiss_path, embeddings)
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