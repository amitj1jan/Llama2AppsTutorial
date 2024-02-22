from langchain.llms import CTransformers
import chainlit as cl

n_gpu_layers = 10
n_batch = 256

config = {'max_new_tokens': 512, 'context_length': 4096,         
            'gpu_layers': n_gpu_layers,'batch_size': n_batch,   
            'temperature': 0.1
         }

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
        config=config,          # Configuration for the model, like context length, max new tokens, temperature etc.
    )
    return(llm) 
