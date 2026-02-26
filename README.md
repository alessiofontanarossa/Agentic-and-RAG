# Agentic AI and RAG

This repository contains applications of RAG and Agentic AI Automation with n8n:

- Complete_RAG (OpenAI): An advanced, modular Agentic RAG application designed for robust document interaction and intelligent information retrieval. Key Features:
    - 🧠 Agentic Backend: Powered by LangChain agents and a Dockerized Qdrant vector database for high-performance semantic search.
    - ✨ Fluid UI: A highly interactive Streamlit frontend that ensures a seamless user experience.
    - 📂 Dynamic Knowledge Base: Users can easily upload new documents, selectively delete files, and instantly reset chat sessions directly from the interface.
    - 🔍 Transparent Citations: The RAG system ensures traceability by always citing exact sources, including the document name and specific page numbers.
    - 🌐 Smart Web Fallback: If the internal database lacks sufficient information, the agent autonomously triggers a web search to find the answer.
    - 🛠️ Real-time Feedback: The UI clearly communicates the agent's thought process, showing the user exactly whether it is retrieving internal data or searching the web.
    
- HuggingFace_RAG: Basic example on the construction of a RAG with HF open-source models. It shows how to combine the user prompt with the retrieved chunks.
- Agno_RAG with memory (OpenAI): Supernotes agent that helps a student in preparing an exam. It employs a team of agents: the leader delegates to a financial expert (which do RAG on the student notes) or to a scraper agent (which search the web if the information are not present in the student notes). There is an in-session memory (memory about user's preferences and chat memory) and an out-session memory, saving the vectorized document and the chat-contents in a vector db. Gradio is used as a front-end interface.
- LangChain_RAG_with_memory.py (OpenAI): How to orchestrate an LLM to do RAG with memory in LangChain. Gradio is used as a front-end interface.
- cv_cl.py (OpenAI): an Agno automation that, given a set of job descriptions, produces 3(2) versions of the curriculum vitae (cover letter), optminizing them on the job keywords using knowledge of the user. At the end, the agent sends an email to the user with the written CV/Cover Letter. Recall to add the Oauth2 autentication for gmail on GCC (Google Cloud Console).
- ollama_RAG.py (Local): a full local RAG system which uses EmbeddingGemma for documents/query embedding and Llama3:2 for generation. It requires Ollama running and a Docker Container with running Qdrant. Streamlit is used as the front-end, which reproduces the main functionalities of the Chat-GPT interface (upload documents, eliminate documents, ecc...)
