import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    PDF_DIRECTORY = "data/raw"
    CHROMA_DB_PATH = "data/chroma_db"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "models/text-embedding-004"
    LLM_MODEL = "gemini-3-flash-preview"

    @classmethod
    def validate(cls):
        if not cls.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        os.makedirs(cls.PDF_DIRECTORY, exist_ok=True)
        os.makedirs(cls.CHROMA_DB_PATH, exist_ok=True)