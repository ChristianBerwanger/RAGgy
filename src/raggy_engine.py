from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from configs.config import Config

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
        self._init_query_rewriter_chain()

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
        retrieval_step = RunnableParallel({
            "docs": retriever,
            "input": RunnablePassthrough()
        })
        generation_step = (
                RunnablePassthrough.assign(
                    context=lambda x: _format_docs(x["docs"])  # Format docs for the prompt
                )
                | prompt
                | self.llm
                | StrOutputParser()
        )
        #self.rag_chain = (
        #    {
        #        "context": retriever | _format_docs,
        #        "input": RunnablePassthrough()
        #    }
        #    | prompt
        #    | self.llm
        #    | StrOutputParser()
        #)
        self.rag_chain_new_pipe = retrieval_step.assign(answer=generation_step)

    def _init_query_rewriter_chain(self):
        rewriter_system_prompt = (
            "Rephrase the user query to optimize it for a vector database retrieval system.\n"
            "Your goal is to rephrase the query inside the <query> tags to be specific, keyword-rich, and distinct.\n"
            "### Isolation Rules:\n"
            "1. **Context Only:** Do not use your internal knowledge or training data."
            "2. Do not use your internal information to interpret this query. \n"
            "3 **Ambiguity:** If the input is ambiguous, do NOT try to resolve this based on internal knowledge or the history.\n"
            "### Instructions:\n"
            "- Remove conversational filler words.\n"
            "- If the query is already optimal, return it exactly as is. \n"
            "- Do NOT answer the question. ONLY return the rewritten query text.\n"
        )
        rewriter_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", rewriter_system_prompt),
                ("human", "<query>\n{input}\n</query>"),
            ]
        )
        self.rewriter_chain = rewriter_prompt | self.llm | StrOutputParser()

    def rewrite_query(self, query: str):
        if not query:
            return ""
        try:
            rewritten = self.rewriter_chain.invoke(query)
            print(f"Rewritten Query: {rewritten}")
            return rewritten
        except Exception as e:
            return


    def ask(self,query: str):
        if not query:
            return "What would you like to know?"
        try:
           # response = self.rag_chain.invoke(self.rewrite_query(query))
           # response_2 = self.rag_chain_new_pipe.invoke(self.rewrite_query(query))['answer']
            response = self.rag_chain_new_pipe.invoke(query)['answer']

            return response
        except Exception as e:
            return f"Error generating response: {e}"

    async def aask(self,query: str):
        if not query:
            return "What would you like to know?"
        try:
           # response = self.rag_chain.invoke(self.rewrite_query(query))
           # response_2 = await self.rag_chain_new_pipe.ainvoke(query)
            response = await self.rag_chain_new_pipe.ainvoke(query)

            return response
        except Exception as e:
            return f"Error generating response: {e}"
