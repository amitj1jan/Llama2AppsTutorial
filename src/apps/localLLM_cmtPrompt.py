from llama_index.llms.llama_utils import messages_to_prompt, completion_to_prompt
from llama_index.llms import LlamaCPP
from datetime import datetime
import sys

# conda environment for this app - 
# conda activate llm
# command to run this app
# python3 src/apps/localLLM_cmtPrompt.py

# how many tokens are processed in parallel.
n_batch = 256

# determines how many layers of the model are offloaded to your GPU.
# Change this value based on your model and your GPU VRAM pool.
n_gpu_layers = 30

# path to offline downloaded llm model 
modelpath = "models/llama-2-7b-chat.Q2_K.gguf"


llm = LlamaCPP(
        model_path = modelpath,
        # this parameter controls the response's factualness to creativity(its range is: 0-1), 
        # i.e. lower the value, more factual response would be and higher the value, more creative a response would be.
        temperature=0.1,
        
        max_new_tokens=3000,         
        # llama2 has a context window of 4096 tokens, but I set it lower to allow for some wiggle room    
        context_window=3900,
    
        # kwargs to pass to __call__()
        generate_kwargs = {},
    
        # kwargs to pass to __init__()
        model_kwargs = {"n_gpu_layers": n_gpu_layers,
                           "n_batch": n_batch},
        
        # transform inputs into Llama2 format
        messages_to_prompt = messages_to_prompt,
        completion_to_prompt=completion_to_prompt,
        verbose=True,
    )


# Start chatting with the chatbot
while True:
    query = input('Prompt: ')
    start_time = datetime.now()
    if query.lower() in ["exit", "quit", "q"]:
        print("Exiting")
        sys.exit()

    # generates the response based on the query provided
    response = llm.complete(query)
    
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("total time taken was:", time_taken)
    print('Answer: ' + response.text + '\n')

