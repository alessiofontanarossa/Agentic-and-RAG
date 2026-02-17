# pip install dotenv wget
# pip install -U langchain
# pip install langchain_community langchain_openai


from dotenv import load_dotenv
# insert in '.env' your OPENAI_API_KEY = '...'
load_dotenv()
import os, wget

##############################################
################ MODELS SETUP ################
##############################################


""" model for embedding: open-source from HuggingFace or OpenAI model"""
# pip install -q langchain-huggingface sentence-transformers
# from langchain_huggingface import HuggingFaceEmbeddings
# embedding_model = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-mpnet-base-v2")

# pip install -q langchain-openai
from langchain_openai import OpenAIEmbeddings
embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small",
                                   dimensions = 1536) # if not specificed, goes to default

len(embedding_model.embed_query("test query")) # 1536

""" model for generation """
from langchain_openai import ChatOpenAI
gpt4o = ChatOpenAI(model_name = "gpt-4o-mini")


######################################################
################ DOCUMENT PREPARATION ################
######################################################


filename = 'companyPolicies.txt'
url = 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/6JDbUb_L3egv_eOkouY71A.txt'

if not os.path.exists(filename):
    wget.download(url, out = filename)

    with open(filename, 'r') as file:
        contents = file.read()
# print(contents[0:100]) # check

""" bring the text contained in 'filename' into 'document', a Document object """
from langchain_community.document_loaders import TextLoader

loader = TextLoader(filename)
document = loader.load()
len(document) # 1
len(document[0].page_content)# 15660

""" proceed to split the 'document' into chunks """
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size = 100,
                    chunk_overlap = 20,
                    length_function = len)

paragraphs = text_splitter.split_documents(document)
len(paragraphs) # 215
paragraphs[1] # page_content='Our Code of Conduct...' metadata={'source': 'companyPolicies.txt'}


##################################################
################ VECTOR DATABASES ################
##################################################


# pip install chromadb faiss-cpu
from langchain_community.vectorstores import FAISS
# import Chroma in older versions: 
# from langchain_community.vectorstores import Chroma
# import Chroma in newer versions:
# pip install -U langchain-chroma
from langchain_chroma import Chroma

""" here we define the vector databases we will use. There are two ways, the first being:
    - chroma_db = Chroma.from_documents(paragraphs, embedding_model)
    - faiss_db = FAISS.from_documents(paragraphs, embedding_model)
    This defines the db directly from the documents, and can have problems if we run again the script. 
    Indeed, the db is completely updated in the case of 'faiss_db', but vectors are appended again for 'chroma_db'.
    Instead, we create before an instance of the db, and then add vectors: """

COLLECTION_NAME = "my_test_collection"
PERSIST_DIRECTORY = "./chroma_db_data"
# create the instance
chroma_db = Chroma(embedding_function = embedding_model, 
                   collection_name = COLLECTION_NAME,
                   # persist_directory = PERSIST_DIRECTORY, # if we want to save the db 
                ) 

try: chroma_db.delete_collection() # clear the db if it already exists
except: pass 

# add vectors
chroma_db = Chroma.from_documents(documents = paragraphs,
                                  embedding = embedding_model,
                                  collection_name = COLLECTION_NAME)
chroma_db._collection.count() # should be equal to len(paragraphs)

# example od retrieving:
query = "What is the goal of the Drug and Alchool Policy of our company?"
answer_chroma_db = chroma_db.similarity_search(query, k = 5)
len(answer_chroma_db) # 5
answer_chroma_db[0] # page_content='Policy Objective: The...' metadata={'source': 'companyPolicies.txt'}

answer_chroma_db = chroma_db.similarity_search_with_score(query, k=5)
doc, score = answer_chroma_db[0]
doc # page_content='Policy Objective: The...' metadata={'source': 'companyPolicies.txt'}
score # 0.7222462892532349


###########################################
################ RETRIEVER ################
###########################################


""" construct a LangChain retriever (with the invoke method) based on the vector store """

chroma_retriever = chroma_db.as_retriever(search_kwargs = {"k": 5},
                                          search_type = "similarity")

query = "What is the goal of the Drug and Alchool Policy of our company?"
chroma_retriever.invoke(query)


#################################################
################ RAG WITH MEMORY ################
#################################################


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {} # this emules a storage for the CHAT HISTORY; use an external db for real use-cases
def get_session_history(session_id: str):
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  return store[session_id]

message = [
    ("system", "You are an useful assistant which helps workers about company rules and values."),
    MessagesPlaceholder(variable_name = "chat_history"), 
    ("user", "Answer to this query ```{query}``` using the context ```{top_context}``` and your memory.")
  ]
prompt = ChatPromptTemplate.from_messages(message)

chain = prompt | gpt4o | StrOutputParser()

chain_with_memory = RunnableWithMessageHistory(
                        chain,
                        get_session_history,
                        history_messages_key = "chat_history",
                        input_messages_key = "query")

def generate_answer(query):
  retrieved = chroma_retriever.invoke(query)
  top_context_list = [retrieved[i].page_content for i in range(len(retrieved))]
  top_context = " ".join(top_context_list)

  return chain_with_memory.invoke({"query": query, "top_context": top_context},
                                  config = {"configurable": {"session_id": "example_session"}})

store # {}
query = "What is the goal of the Drug and Alchool Policy of our company?"
generate_answer(query) # The goal of the Drug and Alcohol Policy of our company is ...

store # {'example_session': InMemoryChatMessageHistory(messages=[HumanMessage(content='What is the goal of the Drug and Alchool Policy of our company?', 
      #                                                                       additional_kwargs={}, response_metadata={}),
      #                                                          AIMessage(content='The goal of the Drug and Alcohol Policy of our company is ...


##################################################
################ GRADIO INTERFACE ################
##################################################


# pip install gradio==5.50.0
import gradio as gr

rag_application = gr.Interface(
    title = "QA Bot",
    description = "Ask any question. The chatbot will try to answer using the provided document.",
    fn = generate_answer,
    allow_flagging = "never",
    inputs = [
        gr.Textbox(label = "Input Query", lines = 20, placeholder = "Type your question here...")
    ],
    outputs = gr.Textbox(label = "Output", lines = 20)
)

theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="indigo",
    neutral_hue="slate",
)

with gr.Blocks(theme=theme, title="QA Bot") as rag_application:
    gr.Markdown(
        """
        # 🤖 QA Bot
        ### Ask any question based on the provided documents.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_box = gr.Textbox(
                label="Input Query",
                placeholder="Type your question here...",
                lines=5,
                autofocus=True
            )
            with gr.Row():
                clear_btn = gr.ClearButton(components=[input_box], variant="secondary")
                submit_btn = gr.Button("Submit Question", variant="primary")

        with gr.Column(scale=1):
            output_box = gr.Textbox(
                label="AI Response",
                lines=10,
                interactive=False,
                show_copy_button=True
            )

    submit_btn.click(
        fn=generate_answer,
        inputs=input_box,
        outputs=output_box
    )

    input_box.submit(
        fn=generate_answer,
        inputs=input_box,
        outputs=output_box
    )
rag_application.launch(share = True)