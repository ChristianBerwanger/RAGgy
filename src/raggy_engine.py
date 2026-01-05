from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from src.config import Config

def _format_docs(docs):
    """Helper to format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

class RAGgy_Engine:
    def __init__(self, vector_store_manager):
        Config.validate()
        self.vector_store_manager = vector_store_manager
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self._init_chain()

    def _init_chain(self):
        """LCEL Pipeline with a custom prompt to prevent using information not provided."""
        system_prompt = (
            "You are an assistant for question-answering tasks.\n"
            "Your goal is to answer user's question based **strictly** "
            "on the provided context below.\n\n"
            "Guidelines:\n"
            "1. **Context Only:** Do not use your internal knowledge or training data."
            "If the answer is not in the context, say that you don't have any information about this.\n"
            "2. **Format:** Answer concise and direct. Do not use phrases like 'Based on the provided text'\n\n"
            
            "<context>\n"
            "{context}\n"
            "</context>"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        retriever = self.vector_store_manager.get_retriever()
        self.rag_chain = (
            {
                "context": retriever | _format_docs,
                "input": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self,query: str):
        if not query:
            return "What would you like to know?"
        try:
            response = self.rag_chain.invoke(query)
            return response
        except Exception as e:
            return f"Error generating response: {e}"
