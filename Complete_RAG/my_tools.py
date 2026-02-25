from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, FewShotPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, CommaSeparatedListOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from qdrant_admin import QdrantAdmin

# can not use a class: the use of @ tool would require the llm to generate a parameter 'self'.
# it is better to use a factory function

def get_my_tools(llm, retriever, subject: str):
    RAG_template = f"""
    You are an expert in {subject} and related topics. Your goal is to provide a highly technical response to the following query:
    ---
    QUERY: {{query}}
    ---
    CONTEXT FROM DATABASE: {{formatted_answer}}
    ---

    INSTRUCTIONS:
    0. Use the same language as the query (if the query is in Italian, answer in Italian, if it's in English, answer in English).
    1. Use ONLY the provided context to answer. If the context is insufficient, state that you lack internal data.
    2. Maintain a professional, high-level financial tone (use technical jargon).
    3. The response must be concise: exactly 5 sentences maximum.
    4. MANDATORY: You must explicitly cite the document name and page number for every claim made.
    5. DO not report the entire piece of document.

    FINAL ANSWER:
    """
    RAG_prompt = PromptTemplate(template = RAG_template, input_variables = ["query", "formatted_answer"])
    RAG_chain = RAG_prompt | llm | StrOutputParser()

    WEB_template = f"""
    You are an expert in {subject} and related topics. Your goal is to provide a highly technical response to the following query:
    ---
    QUERY: {{query}}
    ---
    CONTEXT FROM THE WEB: {{web_results}}
    ---

    INSTRUCTIONS:
    0. Use the same language as the query (if the query is in Italian, answer in Italian, if it's in English, answer in English).
    1. Maintain a professional, high-level financial tone.
    2. The response must be concise: exactly 5 sentences maximum.
    3. MANDATORY: You must explicitly cite the url for every claim made.

    FINAL ANSWER:
    """
    WEB_prompt = PromptTemplate(template = WEB_template, input_variables = ["query", "web_results"])
    WEB_chain = WEB_prompt | llm | StrOutputParser()
    duckduckgo_search_tool = DuckDuckGoSearchRun()

    @tool
    def search_internal_database(query: str) -> str: 
        """ ALWAYS use this tool first for economics questions. 
            Fundamental tool to perform Retrieval Augmented Generation (RAG) on provided documents. """
        retrieved_docs = retriever.invoke(query)
        if not retrieved_docs:
            return "No useful documents found."
        
        formatted_answer = []
        for doc in retrieved_docs:
            page = doc.metadata.get("page", "N/A")
            filename = doc.metadata.get("clean_filename", "Unknown")
            text = doc.page_content
            formatted_answer.append(f"--- FROM FILE: {filename} (Pag: {page}) ---\n{text}")

        context_string = "\n\n".join(formatted_answer)
        return RAG_chain.invoke({"query": query, "formatted_answer": context_string})

    @tool
    def search_internet_duckduckgo(query: str) -> str:
        """ Use this tool ONLY if the 'search_internal_database' fails or the quality of the answer is low.
            This tool enables web search for accurate answers. """
        web_results = duckduckgo_search_tool.invoke(query)
        return WEB_chain.invoke({"query": query, "web_results": web_results})

    return [search_internal_database, search_internet_duckduckgo]