import json
import pandas as pd
import asyncio
from pathlib import Path
from ragas import experiment, Dataset
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import llm_factory
from ragas.embeddings import GoogleEmbeddings
from ragas.metrics.collections import (Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall, AnswerCorrectness)
from google import genai
from src.raggy_engine import RAGgy_Engine
from configs.config import Config
from src.vector_store import VectorStoreManager
'''
TESTED ONLY WITH ONE EXAMPLE -> DATASET GENERATION HITS API LIMITS -> Set Up local LLM
'''
class RAGEvaluator:
    def __init__(self, rag_engine: RAGgy_Engine):
        self.rag_engine = rag_engine
        Config.validate()
        google_client = genai.Client(api_key=Config.GOOGLE_API_KEY)
        self.judge_llm = llm_factory(
            model=Config.JUDGE_LLM_MODEL,
            client=google_client,
            provider="google",
            temperature=0,
            max_tokens=None,
            timeout=None,
            max_retries=3,
            google_api_key=Config.GOOGLE_API_KEY
        )
        self.judge_embeddings = GoogleEmbeddings(
            model=Config.JUDGE_EMBEDDING_MODEL,  # Use the newer 004 model
            client=google_client
        )
        self.faithfulness_scorer = Faithfulness(llm=self.judge_llm)
        self.answer_relevancy_scorer = AnswerRelevancy(llm=self.judge_llm, embeddings=self.judge_embeddings)
        self.context_precision_scorer = ContextPrecision(llm=self.judge_llm)
        self.context_recall_scorer = ContextRecall(llm=self.judge_llm)
        self.answer_correctness_scorer = AnswerCorrectness(llm=self.judge_llm, embeddings=self.judge_embeddings)

    def load_data(self, data_path: str):
        with open(data_path, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} test cases from {data_path}")
        return data

    async def run(self, data_path: str):
        raw_data = self.load_data(data_path)
        dataset = Dataset(
            name="RAG_eval_code_debug_dataset",
            backend="local/csv",
            root_dir="."
        )
        for row in raw_data:
            dataset.append(row)
        @experiment()
        async def evaluate(sample):
            print("TEST!")
            result = await self.rag_engine.aask(sample['user_input'])
            print(result)
            sample['response'] = result['answer']
            sample['retrieved_context'] = [doc.page_contect for doc in result['docs']]

            faithfulness = await self.faithfulness_scorer.ascore(
                user_input=sample['user_input'],
                response=sample['response'],
                retrieved_contexts = sample['retrieved_context']
            )
            answer_relevancy = await self.answer_relevancy_scorer.ascore(
                user_input=sample['user_input'],
                response=sample['response']
            )
            context_precision = await self.context_precision_scorer.ascore(
                user_input=sample['user_input'],
                retrieved_contexts=sample['retrieved_context'],
                reference=sample['label']
            )
            context_recall = await self.context_recall_scorer.ascore(
                user_input=sample['user_input'],
                retrieved_contexts=sample['retrieved_context'],
                reference=sample['label']
            )
            answer_correctness = await self.answer_correctness_scorer.ascore(
                user_input=sample['user_input'],
                response=sample['response'],
                reference=sample['label']
            )
            return {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precisiion": context_precision,
                "conext_recall": context_recall,
                "answer_correctness": answer_correctness
            }
        res = await evaluate.arun(dataset,
                                  name="Initial_test_v1")
        return res


if __name__ == "__main__":
    vm = VectorStoreManager()
    rag = RAGgy_Engine(vm)
    evaluator = RAGEvaluator(rag)
    results = asyncio.run(evaluator.run("eval_data.json"))
    print(results)

