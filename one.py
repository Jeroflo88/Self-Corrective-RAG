from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

DataPath = "data"

# embedding model
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
# read documents
documents = SimpleDirectoryReader(DataPath).load_data()

# chunk docs
parser = SentenceSplitter(chunk_size=128, chunk_overlap=32)
nodes = parser.get_nodes_from_documents(documents)
for node in nodes:
    keys_to_remove = []
    for key, value in node.metadata.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del node.metadata[key]
    

# create Chroma client + collection
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection("my_collection")
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex(
    nodes, storage_context=storage_context, embed_model=embed_model
)

print("DONE!")
