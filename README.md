# Vodafone-rag-chatbot-
Vodafone rag chatbot 
Vodafone Customer Service RAG Chatbot
Developer: Ismail Elsayed
GitHub: https://github.com/ismailelsayedeltanja


Overview
A RAG-based customer service chatbot for Vodafone Egypt. It answers questions about internet plans, calling plans, technical support, and Vodafone Pay by searching a knowledge base and generating responses using Groq.


data/
└── knowledge_base/
    ├── internet_plans.txt
    ├── calling_plans.txt
    ├── technical_support.txt
    └── vodafone_pay.txt 


    
Tech Stack

LLM: Groq API (Llama 3 70B)
Embeddings: HuggingFace sentence-transformers
Vector Store: FAISS
Framework: LangChain
Web UI: Gradio
API: FastAPI


Setup
bashpip install langchain langchain-community faiss-cpu sentence-transformers groq gradio fastapi uvicorn pypdf python-dotenv
Set your Groq API key:
bashexport GROQ_API_KEY="your-key-here"
Get a free key from https://console.groq.com

Run
bashpython vodafone_rag_chatbot.py
To launch Gradio UI, uncomment this line at the bottom of the file:
pythondemo.launch(server_name="0.0.0.0", server_port=7860)
