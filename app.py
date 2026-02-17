import locale
locale.getpreferredencoding = lambda: "UTF-8"

import yaml
import requests
import os
import pandas as pd
import numpy as np
import streamlit as st
import tempfile
from operator import itemgetter

# Load API credentials if available
if os.path.exists("api_keys.yml"):
    with open('api_keys.yml', 'r') as file:
        api_creds = yaml.safe_load(file)
        # read Groq key if present
        GROQ_API_KEY = api_creds.get('GROQ_API_KEY')
    if 'GOOGLE_API_KEY' in api_creds:
        os.environ['GOOGLE_API_KEY'] = api_creds['GOOGLE_API_KEY']
else:
    GROQ_API_KEY = None

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# chromadb not required for in-memory retriever

# Customize initial app landing page
st.set_page_config(page_title="File QA Chatbot", page_icon="🤖")
st.title("Welcome to File QA RAG Chatbot 🤖")

@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    # Read documents
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyMuPDFLoader(temp_filepath)
        docs.extend(loader.load())

    # Split into document chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                   chunk_overlap=200)
    doc_chunks = text_splitter.split_documents(docs)

    # Compute embeddings for chunks and keep in memory (avoid external vector DB)
    # Use free HuggingFace embeddings instead of OpenAI
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    texts = [d.page_content for d in doc_chunks]
    # embed_documents returns list of vectors
    vectors = embeddings_model.embed_documents(texts)
    vectors = np.array(vectors, dtype=float)

    # Return a simple retriever function closure
    def retriever_fn(query, k=3):
        # Compute query embedding with fallbacks and shape safety
        raw_q = embeddings_model.embed_query(query)
        if raw_q is None or (hasattr(raw_q, '__len__') and len(raw_q) == 0):
            try:
                raw_q = embeddings_model.embed_documents([query])[0]
            except Exception:
                raise RuntimeError("Failed to compute query embedding for the given input")
        q_vec = np.array(raw_q, dtype=float)
        # Flatten if nested
        if q_vec.ndim > 1:
            q_vec = q_vec.reshape(-1)

        # Ensure vectors is 2D and non-empty
        if vectors is None or vectors.size == 0:
            raise RuntimeError("Document embeddings are empty; ensure uploaded documents were parsed correctly")
        if vectors.ndim == 1:
            # single vector -> make it a 2D array
            vectors_local = vectors.reshape(1, -1)
        else:
            vectors_local = vectors

        # If q_vec dimensionality doesn't match, try alternative embedding strategies
        if q_vec.size == 0:
            raise RuntimeError("Query embedding is empty; cannot compute similarity")
        if q_vec.shape[0] != vectors_local.shape[1]:
            # Try re-embedding with embed_documents (some HF wrappers return different shapes)
            try:
                alt = embeddings_model.embed_documents([query])[0]
                alt_vec = np.array(alt, dtype=float)
                if alt_vec.ndim > 1:
                    alt_vec = alt_vec.reshape(-1)
                if alt_vec.size == vectors_local.shape[1]:
                    q_vec = alt_vec
                else:
                    raise RuntimeError(f"Embedding dimension mismatch: doc vectors have dim={vectors_local.shape[1]} but query embedding dim={alt_vec.size}")
            except Exception as e:
                raise RuntimeError(f"Embedding dimension mismatch and re-embedding failed: {e}")

        # cosine similarity (safe matmul)
        sims = (vectors_local @ q_vec) / (np.linalg.norm(vectors_local, axis=1) * (np.linalg.norm(q_vec) + 1e-12))
        top_idx = sims.argsort()[-k:][::-1]
        return [doc_chunks[i] for i in top_idx]

    return retriever_fn

# Manages live updates to a Streamlit app's display by appending new text tokens
# to an existing text stream and rendering the updated text in Markdown
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)

# Creates UI element to accept PDF uploads
uploaded_files = st.sidebar.file_uploader(
    label="Upload PDF files", type=["pdf"],
    accept_multiple_files=True,
)
if not uploaded_files:
    st.info("Please upload PDF documents to continue.")
    st.stop()

# Create retriever object based on uploaded PDFs
retriever = configure_retriever(uploaded_files)

# (Using Groq for generation via HTTP)

# Groq generation wrapper (tries common endpoints)
def generate_with_groq(prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
    """Send prompt to Groq HTTP API. Tries a few common endpoints and returns text."""
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in api_keys.yml")
    # Prefer Groq's OpenAI-compatible Responses API (see Groq docs)
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}

    # Try the OpenAI-compatible Responses endpoint first
    try:
        url = "https://api.groq.com/openai/v1/responses"
        payload = {
            "input": prompt,
            "model": "openai/gpt-oss-20b",
            "max_output_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        if not resp.ok:
            try:
                err = resp.json()
                if isinstance(err, dict) and 'error' in err:
                    raise RuntimeError(f"Groq API error: {err['error'].get('message')}")
                raise RuntimeError(f"Groq API error: {resp.status_code}: {resp.text}")
            except ValueError:
                raise RuntimeError(f"Groq API error: {resp.status_code}: {resp.text}")
        data = resp.json()
        # Typical Groq openai-compatible responses have 'output_text' or 'output'
        if isinstance(data, dict):
            if 'output_text' in data:
                if data['output_text'] and data['output_text'].strip():
                    return data['output_text']
                raise RuntimeError('Groq API returned empty output_text; check model access and permissions')
            if 'output' in data and isinstance(data['output'], list):
                texts = []
                for item in data['output']:
                    if isinstance(item, dict):
                        # 'content' or 'text' fields may be present
                        for key in ('content', 'text', 'message'):
                            if key in item:
                                texts.append(item.get(key))
                # filter out empty content
                nonempty = [t for t in texts if t and str(t).strip()]
                if nonempty:
                    return "\n".join(nonempty)
                raise RuntimeError('Groq API returned empty output; the model may be gated or require different model parameter')
            # openai-compatible 'choices' fallback
            if 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
                first = data['choices'][0]
                if isinstance(first, dict):
                    for key in ('text', 'message', 'content'):
                        if key in first:
                            return first.get(key)
            return str(data)
    except Exception:
        pass

    # Fallback: try a generic Groq endpoint pattern (legacy examples)
    endpoints = [
        "https://api.groq.com/v1/generate",
        "https://api.groq.com/v1/completions",
    ]
    payload_generic = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temperature}
    for url in endpoints:
        try:
            resp = requests.post(url, json=payload_generic, headers=headers, timeout=30)
        except Exception:
            continue
        if not resp.ok:
            continue
        try:
            data = resp.json()
        except Exception:
            return resp.text
        if isinstance(data, dict):
            if 'text' in data and isinstance(data['text'], str):
                return data['text']
            if 'choices' in data and isinstance(data['choices'], list) and len(data['choices']) > 0:
                first = data['choices'][0]
                if isinstance(first, dict):
                    for key in ('text', 'message', 'content'):
                        if key in first:
                            return first.get(key)
            return str(data)
    raise RuntimeError('Groq API request failed; check GROQ_API_KEY and endpoint')
# Create a prompt template for QA RAG System
qa_template = """
              Use only the following pieces of context to answer the question at the end.
              If you don't know the answer, just say that you don't know,
              don't try to make up an answer. Keep the answer as concise as possible.

              {context}

              Question: {question}
              """
qa_prompt = ChatPromptTemplate.from_template(qa_template)

# This function formats retrieved documents before sending to LLM
def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

# (We will use a simple in-memory retrieval flow instead of an external vector DB)

# Store conversation history in Streamlit session state
streamlit_msg_history = StreamlitChatMessageHistory(key="langchain_messages")

# Shows the first message when app starts
if len(streamlit_msg_history.messages) == 0:
    streamlit_msg_history.add_ai_message("Please ask your question?")

# Render current messages from StreamlitChatMessageHistory
for msg in streamlit_msg_history.messages:
    st.chat_message(msg.type).write(msg.content)

# Callback handler which does some post-processing on the LLM response
# Used to post the top 3 document sources used by the LLM in RAG response
class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg: st.write):
        BaseCallbackHandler.__init__(self)
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents: # retrieved documents from retriever based on user query
            metadata = {
                "source": d.metadata["source"],
                "page": d.metadata["page"],
                "content": d.page_content[:200]
            }
            idx = (metadata["source"], metadata["page"])
            if idx not in source_ids: # store unique source documents
                source_ids.append(idx)
                self.sources.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if len(self.sources):
            st.markdown("__Sources:__ "+"\n")
            st.dataframe(data=pd.DataFrame(self.sources[:3]),
                          width=1000) # Top 3 sources

# If user inputs a new prompt, display it and show the response
retriever = configure_retriever(uploaded_files)

if user_prompt := st.chat_input():
    st.chat_message("human").write(user_prompt)
    with st.chat_message("ai"):
        # retrieve top documents
        docs = retriever(user_prompt, k=3)
        context = format_docs(docs)
        prompt_text = qa_template.format(context=context, question=user_prompt)
        # Call the LLM (use Groq HTTP API if key present)
        try:
            if GROQ_API_KEY:
                answer_text = generate_with_groq(prompt_text)
                st.write(answer_text)
            else:
                st.error("No Groq API key found in api_keys.yml. Please add GROQ_API_KEY to enable generation.")
        except Exception as e:
            st.error(f"Generation error: {e}")
        # Show sources
        if docs:
            sources = []
            for d in docs:
                sources.append({
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                    "content": d.page_content[:200]
                })
            st.markdown("__Sources:__")
            st.dataframe(data=pd.DataFrame(sources), width=1000)
