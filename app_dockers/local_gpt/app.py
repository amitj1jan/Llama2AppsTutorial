from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain

from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
import chainlit as cl
from datetime import datetime

# command to run this app
# chainlit run app.py --port 8001 -w


prompt_template = """### System Prompt
    You are an helpful AI assistant and your name is SAHAYAK. You are kind, gentle and respectful to the user. Your job is to answer the question sent by the user in concise and step by step manner. If you don't know the answer to a question, please don't share false information.

### Current conversation:
{history}

### Question
{input}

### Assistant"""

# Download the file here https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/blob/main/
# and update the path
MODEL_PATH = "models/model.gguf"


@cl.cache
def instantiate_llm():
    n_batch = (
        256  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.
    )
    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path=MODEL_PATH,
        n_batch=n_batch,
        n_ctx=4096,
        temperature=0.1,
        max_tokens=500,
        callback_manager=callback_manager,
        verbose=True,  # Verbose is required to pass to the callback manager
        streaming=True,
    )
    return llm

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = instantiate_llm()


@cl.on_chat_start
def main():
    prompt = PromptTemplate(template=prompt_template, input_variables=["history", "input"])

    conversation = ConversationChain(
        prompt=prompt, llm=llm, memory=ConversationBufferWindowMemory(k=3)
    )

    cl.user_session.set("conv_chain", conversation)


@cl.on_message
async def main(message: cl.Message):
    start_time = datetime.now()
    conversation = cl.user_session.get("conv_chain")

    answer_prefix_tokens=["FINAL", "ANSWER"]
    cb = cl.LangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=answer_prefix_tokens
    )

    cb.answer_reached = True
    res = await cl.make_async(conversation)(message.content, callbacks=[cb])
    
    end_time = datetime.now()
    time_taken = end_time - start_time
    print("total time taken was:", time_taken)
