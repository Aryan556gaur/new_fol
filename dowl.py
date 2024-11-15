from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings

# Replace with the path where you want to save the model
model_dir = "models/all-MiniLM-L6-v2"

# This will download and save the model locally in the specified directory
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", cache_folder=model_dir)

loader = TextLoader("file.txt")
documents = loader.load()

# Split documents into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = text_splitter.split_documents(documents)

# Load a smaller, faster embedding model
embeddings = HuggingFaceEmbeddings(model_name=r"models\all-MiniLM-L6-v2\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9")

db = FAISS.from_documents(chunked_docs, embeddings)
db.save_local("faiss_index")