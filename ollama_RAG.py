# pip install streamlit agno lancedb qdrant-client ollama pypdf python-docx openai

import shutil, random, os
from tqdm import tqdm

import streamlit as st

from agno.agent import Agent
from agno.models.ollama import Ollama # generation model
from agno.knowledge.embedder.ollama import OllamaEmbedder # embedding model
from agno.knowledge.knowledge import Knowledge # for RAG
from agno.knowledge.reader.pdf_reader import PDFReader # chunking 
from agno.knowledge.reader.text_reader import TextReader
from agno.knowledge.reader.docx_reader import DocxReader
from agno.vectordb.lancedb import LanceDb, SearchType # vector db
from agno.vectordb.qdrant import Qdrant

from qdrant_client import models # filtering, to eliminate a specific document


# ensure Ollama is running and the models are pulled:
#    - 'ollama pull embeddinggemma:latest'
#    - 'ollama pull llama3.2:latest' """

# put all the back-end logic in a function 

COLLECTION_NAME = "docs"
@st.cache_resource # with cache to avoid reloading models and db at each interaction
def get_rag_agent(): 
    ############## MODELS DEFINITION ##############
    embedder_id = "embeddinggemma:latest"
    generator_id = "llama3.2:latest"

    embedder = OllamaEmbedder(id = embedder_id,  dimensions = 768)
    generator = Ollama(id = generator_id)

    ############ KNOWLEDGE DEFINITION ##############
    # vector_db = LanceDb(table_name = "docs",
    #                     uri = "tmp/lancedb_storage", # 'path' if local, 'uri' if not
    #                     search_type = SearchType.vector, # default, usual vector similarity
    #                     embedder = embedder)

    vector_db = Qdrant(collection = COLLECTION_NAME,
                       url = "http://localhost:6333", # since the script is running locally
                       embedder = embedder)

    knowledge = Knowledge(vector_db = vector_db)
    
    # knowledge.insert(path = "...", reader = reader) # insertion left for later

    ############ AGENT DEFINITION ##############
    rag_agent = Agent(
        name = "rag_agent",
        model = generator,
        debug_mode = False,
        description = "You are a pdf expert that reads document and answer questions",
        instructions = [ "Search the knowledge base for relevant information and base your answers on it.",
                        "Be clear, and generate well-structured answers.",
                        "Use clear headings, bullet points, or numbered lists where appropriate.",
                        "CRITICAL: When using the 'search_knowledge_base' tool, you MUST provide ONLY a plain text string as the query.",
                        "DO NOT use JSON, DO NOT use dictionaries, DO NOT include field names like 'value' or 'query'.",
                        "Example of correct tool call: search_knowledge_base('bank transfer')"],
        expected_output = "A clear answer, written in the same language as the question.",
        knowledge = knowledge,
        search_knowledge = True,
        stream = True,
        markdown = True)
    
    return rag_agent

rag_agent = get_rag_agent()

pdf_reader = PDFReader(chunk_size = 1000,
                       split_on_pages = True)
txt_reader = TextReader(chunk_size = 1000,
                        split_on_pages = True)
docx_reader = DocxReader(chunk_size = 1000,
                         split_on_pages = True)

def get_reader(extension = ".pdf"): # '.pdf' cause 'os.path.splitext' gives also the dot '.'
    if extension == ".pdf":
        reader = pdf_reader
    if extension == ".txt":
        reader = txt_reader
    if extension == ".docx":
        reader = docx_reader
    return reader


# front-end configuration 

############ MAIN PAGE and FRONT-END SESSION-STATE ##############
st.set_page_config(page_title = "My local RAG", page_icon = "🤖")
st.title("🤖 Chat with your PDFs (local 😎)")

# we have to sync the database with the session-state at the start of the app
client = rag_agent.knowledge.vector_db.client # acceed to the client and bypass agno from now on
def get_indexed_filenames():
    try:
        # first we have to check that the collection exists
        # if not exists, we return an empty list before scroll
        collections = client.get_collections().collections
        exists = any(c.name == COLLECTION_NAME for c in collections)
        
        if not exists:
            return [] 
        
        points, _ = client.scroll( # this search in a specific collection
            collection_name = COLLECTION_NAME,
            with_payload = True, 
            with_vectors = False, # bypass the 'get' of vectors, faster and lighter
            limit = 1000) # how many points to re-get
        
        # extract the unique filenames
        filenames = list(set([p.payload["name"] for p in points if "name" in p.payload]))
        return filenames
    except Exception as e:
        print(f"Error in loading files from Qdrant: {e}")
        return []
    
if "uploaded_documents" not in st.session_state:
    st.session_state.uploaded_documents = get_indexed_filenames()
            
########### SIDEBAR FOR DOUMENT LOADING ##############
with st.sidebar:
    st.header("📂 Load Document")
    uploaded_file = st.file_uploader("Choose a file:", type = ["pdf", "txt", "docx"]) # upload widget
    
    if uploaded_file is not None:
        filename = uploaded_file.name # complete name, es 'xxx.pdf'
        file_extension = os.path.splitext(uploaded_file.name)[1] # only extension, es '.pdf'
        if not os.path.exists("tmp"):
            os.makedirs("tmp")
        temp_path = f"tmp/{filename}"
        if st.button("🧠 Process file"): # botton to process the uploaded file
            
            if filename in st.session_state.uploaded_documents or filename in os.listdir("tmp"):
                st.warning(f"The file '{filename}' is already in the Knowledge Base!")
            elif filename in get_indexed_filenames():
                st.session_state.uploaded_documents.append(filename)
                st.warning(f"The file '{filename}' is already in the Knowledge Base!")
            else:
                with st.spinner("Processing the file..."):
                    # the following two lines are necessary, because Agno wants local files
                    # but streamlit saves files in the RAM
                    with open(temp_path, "wb") as file:
                        file.write(uploaded_file.getbuffer()) # getbuffer() to avoid copy

                reader = get_reader(extension = file_extension) 
                if reader is not None:
                    rag_agent.knowledge.insert(path = temp_path, reader = reader)
                    st.session_state.uploaded_documents.append(filename)
                    st.success("Document correctly elaborated.")
                    st.rerun()
                else: 
                    st.error("Format not supported")

    ########## DISPLAY KNOWLEDGE BASE ##########
    if st.session_state.uploaded_documents:
        st.markdown("---")
        st.subheader("📚 Knowledge Base")
        for i, doc in enumerate(st.session_state.uploaded_documents):
            st.text(f"{i+1}: {doc}")

        ########## CLEAR ALL DOCUMENTS ##########
        if st.button("🗑️ Clear All"): # this clear all the knowledge base, from database and UI interface
            if os.path.exists("tmp"):
                shutil.rmtree("tmp") # remove the tmp files
            # rag_agent.knowledge.vector_db..clear() # remove from qdrant using agno
            client.delete(collection_name = COLLECTION_NAME,
                          points_selector = models.FilterSelector(
                                filter = models.Filter(must = []) # Filtro vuoto = seleziona tutto
                                ))
            st.session_state.uploaded_documents = [] # remove from UI interface
            st.rerun()
        
        ########## CLEAR A SPECIFIC DOCUMENT ##########
        doc_to_delete = st.selectbox("Select the Document to be Eliminated:", [""] + st.session_state.uploaded_documents)
        
        if doc_to_delete and st.button("❌ Confirm Deletion"):
            with st.spinner(f"Eliminating {doc_to_delete}..."):
                client.delete(collection_name = COLLECTION_NAME,
                              points_selector = models.FilterSelector(
                                filter = models.Filter(
                                    must = [models.FieldCondition(
                                            key = "name", 
                                            match = models.MatchValue(value = doc_to_delete))]
                                )))
                
                file_path = f"tmp/{doc_to_delete}"
                if os.path.exists(file_path):
                    os.remove(file_path)
                
                st.session_state.uploaded_documents.remove(doc_to_delete)
                st.success(f"'{doc_to_delete}' correctly eliminated.")
                st.rerun()
        
########## INPUT MESSAGE INTERFACE ##############
question = st.chat_input("Ask what you want: ")

def stop_gen(): # to stop generation
    st.session_state.stop_generation = True

if question is not None:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        response_container = st.empty() 
        full_response = ""

        st.session_state.stop_generation = False
        st.sidebar.button("🛑 Stop Generation", on_click = stop_gen)

        with st.spinner("Generating the answer..."):
            response_stream = rag_agent.run(question, stream = True)
            
            for chunk in response_stream:

                if st.session_state.stop_generation: # check the interruption of generation
                    full_response += "\n\n**[Generation interrupted by the user]**"
                    break

                if chunk.content:
                    full_response += chunk.content
                    response_container.markdown(full_response + "▌") # update the response container in real-time with a cursor
        
            response_container.markdown(full_response) # remove cursor at the end

            # ############## ADD SOURCES FOR COMPLETENESS ##############
            # if rag_agent.run_response and rag_agent.run_response.sources:
            #     with st.expander("📚 Used Sources for the answer"):
            #         for source in rag_agent.run_response.sources:
            #             source_name = source.document.name if source.document else "Unknown"
            #               
            #             page_num = source.document.meta_data.get("page", "?") if source.document else "?"
            #             st.write(f"- **{source_name}** (Pag. {page_num})")