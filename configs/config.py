import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "models/text-embedding-004"
    LLM_MODEL = "gemini-3-flash-preview"
    DATASETGEN_EMBEDDING_MODEL = "models/text-embedding-004"
    DATASETGEN_LLM_MODEL = "gemini-3-flash-preview"
    JUDGE_EMBEDDING_MODEL = "models/text-embedding-004"
    JUDGE_LLM_MODEL = "gemini-3-flash-preview"
    ROOT_DIR = Path(__file__).resolve().parent.parent
    PDF_DIRECTORY = ROOT_DIR / "data" / "raw"
    CHROMA_DB_PATH = ROOT_DIR /"data" / "chroma_db"

    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        os.makedirs(cls.PDF_DIRECTORY, exist_ok=True)
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)