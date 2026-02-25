# pip install dotenv openai IPython pypdf
# pip install langchain langchain-community langchain_openai langchain_core langgraph==1.0.8 chromadb faiss-cpu langchain-chroma langchain_experimental
# pip install streamlit langchain-qdrant ddgs duckduckgo-search langgraph-checkpoint-sqlite watchdog

#########################################
################ IMPORTS ################
#########################################

# general imports
import os, shutil
from dotenv import load_dotenv
# insert in '.env' your OPENAI_API_KEY = '...' and USER_AGENT = "Supernotes/1.0"
load_dotenv() 
from typing import Literal
from IPython.display import Image, display

# LangChain imports

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, TextLoader, PyMuPDFLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
import faiss
from langchain_community.vectorstores import Chroma, FAISS
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import create_agent
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, Filter, FieldCondition, MatchValue
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

import streamlit as st
from qdrant_admin import QdrantAdmin

from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool

from my_tools import get_my_tools

from IPython.display import Image, display

#################################################
################ HUPERPARAMETERS ################
#################################################

VECTOR_SIZE = 1536 # default of 'text-embedding-3-small'
TEMPERATURE = 0.2
QDRANT_URL = "http://localhost:6333"
USER_NAME = "Alessio"
SUBJECT = "Physics"
QDRANT_COLLECTION_NAME = f"{USER_NAME}_collection_{SUBJECT}"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 5
SHORT_MEMORY_TOKENS = 4000

###################################################
################ BACKEND STRUCTURE ################
###################################################

@st.cache_resource
def backend_setup():

    ##############################################
    ################ MODELS SETUP ################
    ##############################################

    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small",
                                    dimensions = VECTOR_SIZE)                  
    llm = ChatOpenAI(model_name = "gpt-4o-mini",
                    temperature = TEMPERATURE,
                    max_tokens = 2048)

    #################################################
    ################ QDRANT SETTINGS ################
    #################################################

    qdrant_admin = QdrantAdmin(url = QDRANT_URL, 
                        collection_name = QDRANT_COLLECTION_NAME, 
                        vector_size = VECTOR_SIZE)
    qdrant_admin.create_collection() # uses default parameters

    qdrant_db = QdrantVectorStore(
                client = qdrant_admin.client,
                collection_name = QDRANT_COLLECTION_NAME,
                embedding = embedding_model)
    retriever = qdrant_db.as_retriever(
                    search_kwargs = {"k": TOP_K},
                    search_type = "similarity")

    #################################################################
    ################ DOCUMENT CHUNKING and EMBEDDING ################
    #################################################################

    # logically, the ingestion section should be here, but does not require caching

    #####################################################
    ################ TOOLS: TEAM MEMBERS ################
    #####################################################

    search_internal_database, search_internet_duckduckgo = get_my_tools(llm, retriever, SUBJECT)

    all_tools = [search_internal_database, search_internet_duckduckgo]
    llm_with_tools = llm.bind_tools(all_tools)

    #############################################
    ################ TEAM LEADER ################
    #############################################

    TEAM_LEADER_PROMPT = f"""
    ROLE: 
    You are "Supernotes," the personal financial assistant for {USER_NAME}. 
    Your mission is to help the user master the exam materials on {SUBJECT} and related topics.

    BEHAVIORAL INSTRUCTIONS:
    - FIRST INTERACTION: Introduce yourself and greet {USER_NAME} cordially.
    - GREETINGS/THANKS: If {USER_NAME} greets or thanks you, wish her a great day and tell {USER_NAME} is the best.
    - OUTPUT STYLE: Your final response to {USER_NAME} must be a maximum of 10 sentences, maintaining a professional and supportive tone.
    - LANGUAGE: Use for the final answer the same language as in the user query.
    - USER PREFERENCES: You can save user preferences, if useful.
    - REFERENCES: Always cite the source of your information (document name and page number, or url) in the final answer.

    WORKFLOW & DELEGATION LOGIC:
    - PHASE 1 (Core Domain): If the question is about {SUBJECT} or strictly conntected topics (es Physics and Mathematics), or exam material, ALWAYS delegate the query to the 'search_internal_database' tool. Use its output to craft your final response.
    - PHASE 2 (Out of Bounds): If the question is NOT related to {SUBJECT} (e.g., recipes, sports), politely apologize and state that you are not an expert in that field. DO NOT use any tools for these topics.
    - PHASE 3 (Fallback): If you delegated a {SUBJECT} question to the 'search_internal_database' and it returns a poor or inconsistent answer, this indicates the internal database is empty. In this specific case, you MUST delegate the search to the 'search_internet_duckduckgo'."""

    sys_msg = SystemMessage(content = TEAM_LEADER_PROMPT)

    from langchain_core.messages import trim_messages

    trimmer = trim_messages(
        max_tokens = SHORT_MEMORY_TOKENS, 
        strategy = "last", 
        token_counter = llm, 
        include_system = False)

    TEAM_LEADER_NODE_NAME = "team_leader_node"
    def team_leader_node(state: MessagesState):
        short_session_history = trimmer.invoke(state["messages"])
        messages_to_send = [sys_msg] + short_session_history
        return {"messages": [llm_with_tools.invoke(messages_to_send)]}

    ########################################
    ################ MEMORY ################
    ########################################

    # session memory
    # ram_memory_db = MemorySaver()
    import sqlite3
    from langgraph.checkpoint.sqlite import SqliteSaver
    conn = sqlite3.connect("session_history.db", check_same_thread = False)
    persistent_memory_db = SqliteSaver(conn)

    # user memory
    from langgraph.store.memory import InMemoryStore
    user_memory_db = InMemoryStore()

    ###########################################
    ################ WORKFLOW  ################
    ###########################################

    workflow = StateGraph(MessagesState)

    tools_node = ToolNode(tools = all_tools)

    workflow.add_node("Supernotes", team_leader_node)
    workflow.add_node("Team_Members", tools_node)

    workflow.add_edge(START, "Supernotes")
    workflow.add_conditional_edges("Supernotes", tools_condition, {"tools": "Team_Members", END: END})
    workflow.add_edge("Team_Members", "Supernotes")

    financial_assistant_team = workflow.compile(
                checkpointer = persistent_memory_db, 
                store = user_memory_db)
    
    return qdrant_admin, qdrant_db, financial_assistant_team

#################################################################
################ DOCUMENT CHUNKING and EMBEDDING ################
#################################################################

def document_controller(file_path: str) -> bool:
    admissible_extensions = [".pdf", ".txt", ".docx"]
    filename = os.path.basename(file_path)
    extension = os.path.splitext(filename)[1].lower()

    return extension in admissible_extensions

def document_ingestor(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list:
    filename = os.path.basename(file_path)

    if not document_controller(file_path):
        raise ValueError(f"'{filename}' has a non supported extension. Use only one among [.pdf, .txt, .docx] please.")
    
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path)
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)

    file = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size = chunk_size,
                                            chunk_overlap = chunk_overlap,
                                            length_function = len)
    chunks = splitter.split_documents(file)
    return chunks 

def add_documents(qdrant_admin: QdrantAdmin, qdrant_db: QdrantVectorStore, chunks: list):
    normalized_chunks = qdrant_admin.chunks_normalization(chunks)
    qdrant_db.add_documents(normalized_chunks)

########################################################
################ WORKFLOW VISUALIZATION ################
########################################################

# for a .ipynb file
# try:
#     display(Image(financial_assistant_team.get_graph().draw_mermaid_png()))
# except Exception as e:
#     print(f"It has been not possible to draw the workflow graph: {e}")

# for a .py file
# if not os.path.exists("agent_graph.png"):
#     try:
#         png_image = financial_assistant_team.get_graph().draw_mermaid_png()
#         with open("agent_graph.png", "wb") as f:
#             f.write(png_image)
#     except Exception as e:
#         print(f"It has been not possible to draw the workflow graph: {e}")

####################################################
################ Testing db section ################
####################################################

# filepath_1 = "./2601.00162v1.pdf"
# filename_1 = os.path.basename(filepath_1)
# filepath_2 = "./2602.16754v1.pdf"
# filename_2 = os.path.basename(filepath_2)

# chunks_1 = document_ingestor(filepath_1, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
# chunks_2 = document_ingestor(filepath_2, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)

# if not qdrant_admin.is_file_in_db(filename = filename_1):
#     add_documents(qdrant_admin, qdrant_db, chunks = chunks_1)
# if not qdrant_admin.is_file_in_db(filename = filename_2):
#     add_documents(qdrant_admin, qdrant_db, chunks = chunks_2)

# print(qdrant_admin.num_total_points())
# print(qdrant_admin.unique_filenames())

# qdrant_admin.remove_a_file(filename_2)
# print("###############")
# print(qdrant_admin.num_total_points())
# print(qdrant_admin.unique_filenames())

# qdrant_admin.delete_collection(QDRANT_COLLECTION_NAME)