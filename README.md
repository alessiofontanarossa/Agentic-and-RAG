# Agentic AI and RAG

This repository contains applications of RAG and Agentic AI Automation with n8n:

- HuggingFace_RAG: Basic example on the construction of a RAG with HF open-source models. It shows how to combine the user prompt with the retrieved chunks.
- Agno_RAG with memory (OpenAI): Supernotes agent that helps a student in preparing an exam. It employs a team of agents: the leader delegates to a financial expert (which do RAG on the student notes) or to a scraper agent (which search the web if the information are not present in the student notes). There is an in-session memory (memory about user's preferences and chat memory) and an out-session memory, saving the vectorized document and the chat-contents in a vector db. Gradio is used as a front-end interface.
- LangChain_RAG with memory (OpenAI): How to orchestrate an LLM to do RAG with memory in LangChain. Gradio is used as a front-end interface.
- cv_cl.py (OpenAI): an Agno automation that, given a set of job descriptions, produces 3(2) versions of the curriculum vitae (cover letter), optminizing them on the job keywords using knowledge of the user. At the end, the agent sends an email to the user with the written CV/Cover Letter. Recall to add the Oauth2 autentication for gmail on GCC (Google Cloud Console).
- ollama_RAG.py (Local): a full local RAG system which uses EmbeddingGemma for documents/query embedding and Llama3:2 for generation. It requires Ollama running and a Docker Container with running Qdrant. Streamlit is used as the front-end, which reproduces the main functionalities of the Chat-GPT interface (upload documents, eliminate documents, ecc...)
