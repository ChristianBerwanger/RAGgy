import json
import pandas as pd
from typing import List, Dict, Optional
import glob
from pathlib import Path
from configs.config import Config

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer
)
'''
NOT FULLY TESTED. TOO MANY API CALLS WHILE CREATING -> Set Up local LLM
'''

class SynthDataGenerator:
    def __init__(self):
        Config.validate()
        google_llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL,
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
            google_api_key=Config.GOOGLE_API_KEY
        )
        google_embeddings=GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL
        )
        self.generator_llm = LangchainLLMWrapper(google_llm)
        self.generator_embeddings = LangchainEmbeddingsWrapper(google_embeddings)
        self.generator = TestsetGenerator(
            llm=self.generator_llm,
            embedding_model=self.generator_embeddings,

        )
        self.documents = []

    def load_documents(self, docs_dir):
        loader = DirectoryLoader(docs_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
        raw_docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=Config.CHUNK_SIZE, chunk_overlap=Config.CHUNK_OVERLAP)
        self.documents = splitter.split_documents(raw_docs)

    def generate(self, test_size: int = 10):
        distributions = [
            (SingleHopSpecificQuerySynthesizer(llm=self.generator_llm), 0.5),
            (MultiHopAbstractQuerySynthesizer(llm=self.generator_llm), 0.3),
            (MultiHopSpecificQuerySynthesizer(llm=self.generator_llm), 0.2),
        ]
        dataset = self.generator.generate_with_langchain_docs(
            self.documents,
            testset_size=test_size,
            query_distribution=distributions
        )
        return dataset.to_pandas()

    def save_to_json(self, df: pd.DataFrame, output_path: str = "eval_data.json"):
        export_data = []
        for _, row in df.iterrows():
            export_data.append({
                "user_input": row['user_input'],
                "reference": row['reference'],
                "retrieved_contexts": row.get('reference_contexts', [])  # Safe access
            })

        with open(output_path, "w", encoding='utf-8') as f:
            json.dump(export_data, f, indent=4)


if __name__ == "__main__":
    gen = SynthDataGenerator()
    gen.load_documents(Config.ROOT_DIR/"data"/"raw")
    df = gen.generate(test_size=5)
    gen.save_to_json(df)