from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core import VectorStoreIndex
import nest_asyncio
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import streamlit as st
import os
nest_asyncio.apply()
os.environ["HF_TOKEN"] = st.secrets["HF_TOKEN"]
qdrant_key = st.secrets["qdrant"]

Settings.llm = Groq(model="llama3-8b-8192", api_key="")
Settings.embed_model = HuggingFaceEmbedding(model_name="law-ai/CustomInLawBERT", trust_remote_code = True)

qdrant_client = QdrantClient(
    url="https://a93013fc-2adb-4ba0-a6a1-23d524b33f9b.europe-west3-0.gcp.cloud.qdrant.io:6333", 
    api_key=qdrant_key,
)

@st.cache_resource(show_spinner=False)
def load_data():
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name="legal_v1", enable_hybrid=True)
    return VectorStoreIndex.from_vector_store(vector_store=vector_store)

@st.cache_resource(show_spinner=False)
def load_retriver():
    return VectorIndexRetriever(
    index=index,
    similarity_top_k=5,
    sparse_top_k=5,
    vector_store_query_mode="hybrid"
    )

index = load_data()

retriever = load_retriver()

st.title("Legal Documents Hybrid Search")

search = st.text_input("Search through documents by keyword", value="")

search_btn = st.button("Search")

if search_btn:
    nodes = retriever.retrieve(search)
    st.write(len(nodes))
    for node in nodes:
        st.write(node.metadata['file_name'])
        st.write("Node-id = " + node.node_id)
        st.write("Text = " +node.text)
        st.write("---")
