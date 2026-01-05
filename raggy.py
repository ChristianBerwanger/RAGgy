import streamlit as st
import os
import tempfile
from src.vector_store import VectorStoreManager
from src.raggy_engine import RAGgy_Engine

st.set_page_config(page_title="RAGgy", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Cache DB/LLM to not reload these everytime
@st.cache_resource
def get_managers():
    vm = VectorStoreManager()
    rag = RAGgy_Engine(vm)
    return vm, rag

vector_store_manager, raggy_engine = get_managers()
st.title("RAGgy")

if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type="pdf",
    key=f"uploader_{st.session_state['file_uploader_key']}"
)
if uploaded_file is not None:
    if uploaded_file.name not in st.session_state.processed_files:

        with st.spinner("Processing PDF..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            state, msg = vector_store_manager.add_pdf(tmp_path, uploaded_file.name)
            os.remove(tmp_path)

            if state == 0:
                st.session_state.processed_files.append(uploaded_file.name)
                st.rerun()
            else:
                st.error(msg)
    else:
        st.toast(f"File '{uploaded_file.name}' added.")


with st.sidebar:
    st.header("Documents")
    files = vector_store_manager.list_pdfs()
    if files:
        for f in files:
            col1, col2 = st.columns([4,1])
            col1.text(f)
            if col2.button("X", key=f"delete_{f}", help=f"Deletes {f}"):
                state, msg = vector_store_manager.delete_pdf(f)
                if state == 0:
                    st.toast(msg)
                    st.rerun()
                else:
                    st.error(msg)
    else:
        st.info("No documents found.")

# Display History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("What's your Question?"):
    # User message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = raggy_engine.ask(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

