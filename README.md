# Llama2Apps
Custom generativeAI Solutions using Llama-2

## Overview
This project utilizes the powerful Llama2 model from Meta to create a collection of cool applications using LangChain, ChainLit, and Llama Index. These applications serve various purposes, offering diverse user experiences.

## Features
- **Command Prompt Chatbox:** A chatbox application using the command prompt interface, providing a simple and efficient way to interact with the Llama2 model.

- **Chatbox with GUI:** Explore the capabilities of Llama2 through a user-friendly ChainLit interface, making conversations more interactive and engaging.

- **Context-Aware Chatbox with GUI:** Takes the user-friendly ChainLit interface experience a step further by being context-aware, making conversations more interactive and engaging.

- **RAG Implementation Chatbox:** Incorporates the Retrieve-and-Generate (RAG) approach to enhance the chatbox experience, generating responses based on relevant retrieved information.

- **Context-Aware RAG Chatbox:** Takes the RAG implementation a step further by being context-aware, creating more coherent and contextually relevant responses.

## Technologies Used
- **Llama2 Model:** The core language model from Meta, providing state-of-the-art natural language processing capabilities.

- **LangChain:** Utilized for seamless integration and communication with the Llama2 model.

- **ChainLit:** Employs a user-friendly interface for enhanced user interactions with the Llama2 model.

- **Llama Index:** Enhances search and retrieval capabilities for better performance in applications like the RAG implementation.

## Getting Started
### Prerequisites
- This project has been developed using python=3.10
### Installation
1. Clone the git repo
   cd Llama2Apps   
   pip install -r requirements.txt
### Usage
1. How to run the app:
```
# How to run chatbox with command prompt
python3 src/apps/localLLM_cmtPrompt.py
```

```
# How to run chatbox with GUI interface
chainlit run src/apps/localLLM_withRAG.py
```

### License
This project is licensed under the MIT License.

### Contact
- Amit Kumar Jha
- amitj1jan@gmail.com
