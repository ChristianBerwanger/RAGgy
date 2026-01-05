from pathlib import Path
from typing import List, Union

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import Config

class VectorStoreManager:
    def __init__(self):
        Config.validate()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=Config.EMBEDDING_MODEL
        )
        self.vector_store = Chroma(
            collection_name="Knowledge_Base",
            embedding_function=self.embeddings,
        persist_directory=Config.CHROMA_DB_PATH
        )

    def add_pdf(self, file_path: Union[str, Path], original_filename: str):
        """
        Processes PDF and stores it in the vector store.
        """
        try:
            path = Path(file_path)
            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = original_filename # Important for Deletion of files
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=Config.CHUNK_SIZE,
                chunk_overlap=Config.CHUNK_OVERLAP
            )
            splits = text_splitter.split_documents(docs)
            self.vector_store.add_documents(documents=splits)
            return 0, f"Successfully added {original_filename} ({len(splits)} chunks)."
        except Exception as e:
            return -1, str(e)

    def list_pdfs(self) -> List[str]:
        """
        Lists all unique PDF Filenames currently in the database.
        :return: List of PDF names (Strings).
        """
        try:
            data = self.vector_store.get(include=['metadatas'])
            metadatas = data.get('metadatas', [])
            files = set()
            for meta in metadatas:
                if meta and "source" in meta:
                    files.add(meta["source"])
            return list(files)
        except Exception as e:
            print(f"Error listing files: {e}")
            return []

    def delete_pdf(self, file_name: str):
        """
        Deletes all chunks associated with a specific filename.
        """
        try:
            data = self.vector_store.get(where={"source": file_name})
            ids_to_delete = data.get('ids', [])
            if not ids_to_delete:
                return -1, "File not found in database."
            self.vector_store.delete(ids=ids_to_delete)
            return 0, f"Deleted {file_name}."
        except Exception as e:
            return -1, str(e)

    def get_retriever(self):
        return self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 5}
        )
