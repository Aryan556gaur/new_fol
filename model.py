from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from flask import Flask, request, jsonify

app = Flask(_name_)

# Load documents and create embeddings 
loader = TextLoader("file.txt")
documents = loader.load()

# Split documents into smaller chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunked_docs = text_splitter.split_documents(documents)

# Load a smaller, faster embedding model
embeddings = HuggingFaceEmbeddings(model_name=r"models\all-MiniLM-L6-v2\models--sentence-transformers--all-MiniLM-L6-v2\snapshots\fa97f6e7cb1a59073dff9e6b13e2715cf7475ac9")

db = FAISS.load_local('faiss_index',embeddings, allow_dangerous_deserialization=True)

# Function to retrieve relevant document based on query
def get_relevant_document(query):
    retriever = db.as_retriever()
    relevant_document = retriever.invoke(query)  # Synchronous retrieval instead of async
    return relevant_document

# Function to get the response from Ollama LLM
def get_response(user_query, relevant_document):
    # Correctly initialize OllamaLLM with the required model configuration
    model = OllamaLLM(model = "llama2")  # Pass the model argument as a dictionary
    
    prompt = f'give answer to {user_query} based on {relevant_document}'  # Combine query and document for context
    
    # Get the response from the model
    response = model.generate([prompt])
    
    return response

# Main function to run the entire process
def main():
    user_query = input("Ask question...?")  # Capture user input
    relevant_document = get_relevant_document(user_query)
    
    if relevant_document:
        response = get_response(user_query, relevant_document)
        response_text = response.generations if response.generations else "No response generated."
        print(response)
    else:
        print("No relevant document found.")

@app.route('/query', methods=['POST'])
def query():
    user_query = request.json['query']
    relevant_document = get_relevant_document(user_query)

    if relevant_document:
        response = get_response(user_query, relevant_document)
        response_text = response.generations[0].text if response.generations else "No response generated."
        return jsonify({'response': response_text})
    else:
        return jsonify({'response': 'No relevant document found.'})

if _name_ == '_main_':
    app.run(debug=True)