from llama_index.core.response.notebook_utils import display_source_node
# from llama_index.core.retrievers import QueryFusionRetriever
# from llama_index.retrievers.bm25 import BM25Retriever
import nest_asyncio
# from llama_index.vector_stores.chroma import ChromaVectorStore
# from llama_index.core.storage.docstore import SimpleDocumentStore
# from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.core import Settings
import streamlit as st
import os
nest_asyncio.apply()

os.environ["HF_TOKEN"] = st.secret["HF_TOKEN"]

Settings.llm = Groq(model="llama3-8b-8192", api_key="")
Settings.embed_model = HuggingFaceEmbedding(model_name="dunzhang/stella_en_400M_v5", trust_remote_code = True)

# db = chromadb.PersistentClient(path="./legal_doc_hybrid_v2")
# chroma_collection = db.get_or_create_collection("dense_vectors")
# vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# docstore = SimpleDocumentStore.from_persist_path("docstore.json")

# storage_context = StorageContext.from_defaults(
#     docstore=docstore, vector_store=vector_store
# )
# index = VectorStoreIndex(nodes=[], storage_context=storage_context)

# retriever = QueryFusionRetriever(
#     [
#         index.as_retriever(similarity_top_k=5),
#         BM25Retriever.from_defaults(
#             docstore=index.docstore, similarity_top_k=5
#         ),
#     ],
#     num_queries=1,
#     use_async=True,
#     retriever_weights=[0.4, 0.6],
#     similarity_top_k=5,
#     mode="relative_score",
#     verbose=True,
# )

# Show title and description.
st.title("Legal Documents Hybrid Search")

search = st.text_input("Search through documents by keyword", value="")

mod = HuggingFaceEmbedding(model_name="dunzhang/stella_en_400M_v5", trust_remote_code = True)

if st.button("Search"):
    embedding = mod.get_text_embedding(search)
    st.write(len(embedding))
    # nodes = retriever.retrieve(search)
    # for node in nodes:
    #     st.write(node.metadata['file_name'])
    #     # print("---")
    #     st.write(node)
