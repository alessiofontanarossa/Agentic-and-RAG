import streamlit as st
import os, shutil, time
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
import uuid

# put main page here before loading heavy stuff from backend
st.set_page_config(page_title = "My RAG app", page_icon = "🤖", layout = "wide")
st.title("🤖 Chat with your PDFs")

from backend import backend_setup, document_ingestor, add_documents
qdrant_admin, qdrant_db, financial_assistant_team = backend_setup()

# import also the useful hyperparameters that are needed in the front-end
from backend import QDRANT_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP, USER_NAME, SUBJECT

# fundamental components of streamlit
# st.chat_message("user" / "assistant")
# st.chat_input("Write here...")
# st.write()
# st.sidebar
# st.file_uploader
# st.columns([1, 2, 1])
# st.expander("Click to expand")
# st.tabs(["Chat", "Documents", "Settings"])
# st.button("Upload") # return True
# st.file_uploader
# st.toggle / st.checkbox
# st.selectbox
# st.write() / st.markdown()
# st.spinner("Uploading...")
# st.toast("File saved!")
# st.success() / st.error() / st.warning()

#######################################################################
################ MAIN PAGE and FRONT-END SESSION-STATE ################
#######################################################################

# when the app starts, we have to control which documents are already ingested and sync them with the session
def get_initial_files():
    if not qdrant_admin.exists_collection(QDRANT_COLLECTION_NAME):
        return []
    return qdrant_admin.unique_filenames()

if "uploaded_files" not in st.session_state:
    try:
        st.session_state["uploaded_files"] = get_initial_files()
    except Exception as e:
        st.error(f"🚨 It is not possible to connect to the Qdrant database:\n\n{e}\n\n We will continue with an empty database.")
        st.session_state["uploaded_files"] = []

##############################################################
################ SIDEBAR FOR DOCUMENT LOADING ################
##############################################################

with st.sidebar:

    ########## DOCUMENT LOADING ##########

    st.header("📂 Load Document")
    uploaded_file = st.file_uploader("Choose a file:", type = ["pdf", "txt", "docx"]) # upload widget
    
    if uploaded_file is not None:
        filename = uploaded_file.name # complete name, es 'xxx.pdf'
        file_extension = os.path.splitext(uploaded_file.name)[1] # only extension, es '.pdf'
        admissible_extensions = [".pdf", ".txt", ".docx"]
        if file_extension not in admissible_extensions:
            st.error(f"🚨 The extension of '{filename}' is not allowed (use only {admissible_extensions})")
        
        if st.button(f"🧠 Process '{filename}'"): # botton to process the uploaded file

            if filename in st.session_state["uploaded_files"] or qdrant_admin.is_file_in_db(filename = filename):
                st.warning(f"The file '{filename}' is already in the Knowledge Base!")
            else:
                with st.spinner("Processing the file..."):
                    if not os.path.exists("tmp"):
                        os.makedirs("tmp")
                    temp_path = f"tmp/{filename}"
                    with open(temp_path, "wb") as file:
                        file.write(uploaded_file.getbuffer())
                    try:
                        chunks = document_ingestor(temp_path, chunk_size = CHUNK_SIZE, chunk_overlap = CHUNK_OVERLAP)
                        if not qdrant_admin.is_file_in_db(filename = filename):
                            add_documents(qdrant_admin, qdrant_db, chunks)
                        st.session_state["uploaded_files"].append(filename)
                        st.success("Document correctly elaborated.")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")
                    finally:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                time.sleep(1)
                st.rerun()
           
    ########## DISPLAY KNOWLEDGE BASE ##########
    if st.session_state["uploaded_files"]:
        st.markdown("---")
        st.subheader("📚 Knowledge Base")
        for i, doc in enumerate(st.session_state["uploaded_files"]):
            st.text(f"{i+1}: {doc}")
    
        ########## CLEAR ALL DOCUMENTS ##########
        if st.button("🗑️ Clear All"): # this clear all the knowledge base, from database and UI interface
            try:
                if os.path.exists("tmp"): # 1) remove the tmp files, even if already done before
                    shutil.rmtree("tmp") 
                for filename in st.session_state["uploaded_files"]:
                    qdrant_admin.remove_a_file(filename) # 2) remove the entire collection from the db
                st.session_state["uploaded_files"] = [] # 3) remove from UI interface
                st.success("All files removed from the Knowledge Base.")
            except Exception as e:
                st.error(f"🚨 The following error has been raised:\n\n{e}")
            
            time.sleep(1)
            st.rerun()

        ########## CLEAR A SET OF DOCUMENTS ##########
        if len(st.session_state["uploaded_files"]) > 1:
            filename_to_delete = st.selectbox("Select the Document to be Eliminated:", [""] + st.session_state["uploaded_files"])
                
            if filename_to_delete and st.button("❌ Confirm Deletion"):
                with st.spinner(f"Eliminating {filename_to_delete}..."):
                    try:
                        if os.path.exists(f"tmp/{filename_to_delete}"): # 1) remove the tmp file, even if already done before
                            os.remove(f"tmp/{filename_to_delete}") 

                        qdrant_admin.remove_a_file(filename_to_delete) # 2) remove the single from the db
                        st.session_state["uploaded_files"].remove(filename_to_delete)# 3) remove from UI interface
                        st.success(f"File '{filename_to_delete}' removed from the Knowledge Base.")
                    except Exception as e:
                        st.error(f"🚨 The following error has been raised:\n\n{e}")
                
                time.sleep(1)
                st.rerun()
        
    ########## CLEAN CHAT ##########
    st.markdown("---")
    st.subheader("💬 Chat Management")
    if st.button("🧹 Clean chat"):
        new_id = uuid.uuid4().hex[:8]
        new_thread_id = f"{USER_NAME}_session_{new_id}"

        st.session_state["thread_id"] = new_thread_id
        st.query_params["thread_id"] = new_thread_id

        st.success(f"✔︎ Chat cleaned.")
        time.sleep(2)
        st.rerun() 

################################################
################ CHAT INTERFACE ################
################################################  

query_params = st.query_params

if "thread_id" in query_params:
    st.session_state["thread_id"] = query_params["thread_id"]
elif "thread_id" not in st.session_state:
    default_id = f"{USER_NAME}_session"
    st.session_state["thread_id"] = default_id
    st.query_params["thread_id"] = default_id

config = {"configurable": {"thread_id": st.session_state["thread_id"]}}

# this first step is necessary to reproduce the 'chat effect'
try:
    current_state = financial_assistant_team.get_state(config)
    if "messages" in current_state.values:
        for msg in current_state.values["messages"]:
            if isinstance(msg, HumanMessage):
                st.chat_message("user").write(msg.content)
            elif isinstance(msg, ToolMessage):
                with st.chat_message("assistant"):
                    st.markdown(f"📄 **Retrieved Documents (via {msg.name}):**\n\n{msg.content}")
            elif isinstance(msg, AIMessage) and msg.content:
                with st.chat_message("assistant"):
                    st.markdown(f"🤖 **Final Answer:**\n\n{msg.content}")
except Exception:
    pass # No past conversation

query = st.chat_input("Ask what you want: ")

if query:
    st.chat_message("user").write(query)
    
    final_answer_container = None
    full_final_answer = ""
    
    with st.spinner("Supernotes is thinking..."):
        for msg_chunk, metadata in financial_assistant_team.stream(
            {"messages": [HumanMessage(content = query)]}, 
            config = config,
            stream_mode = "messages"):
            
            node_name = metadata.get("langgraph_node", "")
            
            if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls and msg_chunk.tool_calls[0]['name'] != "":
                st.toast(f"🛠️ I am using the tool: '{msg_chunk.tool_calls[0]['name']}'")
            
            elif isinstance(msg_chunk, ToolMessage):
                with st.chat_message("assistant"):
                    st.markdown(f"📄 **Retrieved Documents (via {msg_chunk.name}):**\n\n{msg_chunk.content}")
            
            elif isinstance(msg_chunk, AIMessage) and msg_chunk.content:
                if node_name == "Supernotes":
                    if final_answer_container is None:
                        chat_msg = st.chat_message("assistant")
                        final_answer_container = chat_msg.empty()
                    
                    full_final_answer += msg_chunk.content
                    
                    final_answer_container.markdown(f"🤖 **Final Answer:**\n\n{full_final_answer} ▌")
        
        if final_answer_container is not None:
            final_answer_container.markdown(f"🤖 **Final Answer:**\n\n{full_final_answer}")

