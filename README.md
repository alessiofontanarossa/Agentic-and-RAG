# Agentic AI and RAG

This repository contains applications of RAG and Agentic AI Automation with n8n:

- HuggingFace_RAG: Basic example on the construction of a RAG with HF open-source models. It shows how to combine the user prompt with the retrieved chunks.
- Agno_RAG with memory (OpenAI): Supernotes agent that helps a student in preparing an exam. It employs a team of agents: the leader delegates to a financial expert (which do RAG on the student notes) or to a scraper agent (which search the web if the information are not present in the student notes). There is an in-session memory (memory about user's preferences and chat memory) and an out-session memory, saving the vectorized document and the chat-contents in a vector db. Gradio is used as a front-end interface.
- LangChain_RAG with memory (OpenAI): How to orchestrate an LLM to do RAG with memory in LangChain. Gradio is used as a front-end interface.

