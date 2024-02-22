from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain import PromptTemplate

prompt_template = """You are an helpful AI assistant and your name is SAHAYAK. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in concise and step by step manner. 
If you don't know the answer to a question, please don't share false information.
            
Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""

def create_prompt(prompt_template):
    """
    Prompt template for QA retrieval for each vectorstore

    Parameters:
    - prompt_template (str): Defines the characteristics of the chatbox using custom prompt

    Returns:
    - str: Prompt to be used for the chatbox.
    """
    prompt = PromptTemplate(template=prompt_template,
                           input_variables=['context', 'question'])
    return(prompt)
    

def get_response_from_qa_chain(llm, prompt, db):
    """
    Takes the llm model, user prompt, and a VectorStore database, and provides a retrieval QA chain object.

    Parameters:
    - llm (llama2): Open Source large language model (replace with the specific type or module if available)
           
    - prompt (str): User prompt to be provided to the llm
    - db (faiss): VectorStore database or retriever object used in the QA chain
    
    Returns:
    - retrieval_chain: Retrieval QA chain object with source documents (type might be specific to your implementation)
    """
    # Creating a retrieval QA chain using the provided llm, chain type, retriever, and prompt
    retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Replace with the actual chain type
        retriever=db.as_retriever(search_kwargs={'k': 1}),  # Assuming db is used as a retriever
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return(retrieval_chain)

    
def answering_bot(llm, db_faiss_path, embeddings):
    """
    Generates a response using a language model and a Faiss database.

    Parameters:
    - llm (llama2): Large language model instance 
    - db_faiss_path (str): Path to the Faiss database used for retrieval

    Returns:
    - response: Response generated using the language model and Faiss database
    """    
    # Load Faiss database with embeddings
    vectorstore = FAISS.load_local(db_faiss_path, embeddings)
    
    # Create a prompt for the QA chain
    message_prompt = create_prompt(prompt_template)
    
    # Get the response using the QA chain
    response = get_response_from_qa_chain(llm, message_prompt, vectorstore)
    
    return(response)  # Return the generated response
