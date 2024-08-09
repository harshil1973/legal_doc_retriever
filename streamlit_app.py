from llama_index.core.response.notebook_utils import display_source_node
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.retrievers.bm25 import BM25Retriever
import nest_asyncio
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import VectorStoreIndex, StorageContext
import sys
import streamlit as st
import os
nest_asyncio.apply()
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

db = chromadb.PersistentClient(path="./legal_doc_hybrid_v2")
chroma_collection = db.get_or_create_collection("dense_vectors")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

docstore = SimpleDocumentStore.from_persist_path("docstore.json")

storage_context = StorageContext.from_defaults(
    docstore=docstore, vector_store=vector_store
)
index = VectorStoreIndex(nodes=[], storage_context=storage_context)

retriever = QueryFusionRetriever(
    [
        index.as_retriever(similarity_top_k=5),
        BM25Retriever.from_defaults(
            docstore=index.docstore, similarity_top_k=5
        ),
    ],
    num_queries=1,
    use_async=True,
    retriever_weights=[0.4, 0.6],
    similarity_top_k=5,
    mode="relative_score",
    verbose=True,
)

# Show title and description.
st.title("Legal Documents Hybrid Search")

os.environ["HF_TOKEN"] = st.secret["HF_TOKEN"]

search = st.text_input("Search through documents by keyword", value="")

if st.button("Search"):
    nodes = retriever.retrieve("bail application  ")
    for node in nodes:
        st.write(node.metadata['file_name'])
        # print("---")
        st.markdown(node)
