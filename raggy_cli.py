"""
RAGgy CLI - Mainly for debugging backend.
"""
import os
import sys
from pathlib import Path

from src.vector_store import VectorStoreManager
from src.raggy_engine import RAGgy_Engine

def main():
    print("--- Initialite RAGgy CLI ---")
    vm = VectorStoreManager()
    rag = RAGgy_Engine(vm)
    RAW_DATA_DIR = Path("data") / "raw"
    if RAW_DATA_DIR.exists():
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        print("\n"+"="*40)
        print(f"       RAGgy CLI (Source: {RAW_DATA_DIR})")
        print("=" * 40)
        print("1. Ingest all PDFs from data/raw/")
        print("2. List Ingested Documents")
        print("3. Delete a Document")
        print("4. Ask Question")
        print("q. Exit")

        menu_selection = input("\nEnter choice: ").strip().lower()

        match menu_selection:
            case '1':
                # Ingest all PDFs from data/raw
                pdf_files = list(RAW_DATA_DIR.glob("*.pdf"))
                existing_docs = vm.list_pdfs()
                for file_path in pdf_files:
                    filename = file_path.name
                    if filename in existing_docs:
                        continue
                    state, msg = vm.add_pdf(file_path, filename)
                    if state == 0:
                        print("Done")
                    else:
                        print(f"Error: {msg}")
            case '2':
                # List all Docs
                files = vm.list_pdfs()
                if files:
                    for i, f in enumerate(files, 1):
                        print(f"{i}. {f}")
                else:
                    print("No documents found in Vector Store.")
            case '3':
                # Delete a Document from Vector Space
                files = vm.list_pdfs()
                if not files:
                    print("No files to delete.")
                    continue

                print("\n--- Delete Document ---")
                for i, f in enumerate(files, 1):
                    print(f"{i}. {f}")

                selection = input("Enter number to delete: ").strip()

                if selection.isdigit():
                    idx = int(selection) - 1
                    if 0 <= idx < len(files):
                        target_file = files[idx]
                        try:
                            state, msg = vm.delete_pdf(target_file)
                            print(f"Result: {msg}")
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        print("Invalid number.")
                else:
                    print("Invalid input.")
            case '4':
                # Ask a Question
                prompt = input("\n Question: ")
                if prompt:
                    print("\nThinking...")
                    try:
                        # BREAKPOINT HERE
                        response = rag.ask(prompt)
                        print(f"\nRAGgy: {response}\n")
                    except Exception as e:
                        print(f"\nError: {e}")

            case 'q' | 'exit' | 'quit':
                # Exit
                print("Bye!")
                break
            case _:
                # Default
                print("Instalid command. Please try again.")

if __name__ == "__main__":
    main()