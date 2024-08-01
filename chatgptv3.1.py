import os
import sys
import openai
from datetime import datetime
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_chroma import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

embedding = OpenAIEmbeddings()

# Function to reload vectorstore with updated data
def reload_index():
    if PERSIST and os.path.exists("persist"):
        print("Reusing index...\n")
        vectorstore = Chroma(persist_directory="persist", embedding_function=embedding)
    else:
        loader = DirectoryLoader("data/")
        text_loader = TextLoader("data/convo_training_data.txt")
        vectorstore_kwargs = {"persist_directory": "persist"} if PERSIST else {}
        index_creator = VectorstoreIndexCreator(vectorstore_cls=Chroma, vectorstore_kwargs=vectorstore_kwargs, embedding=embedding)
        index = index_creator.from_loaders([loader, text_loader])
        vectorstore = index.vectorstore
        print("data tained!")
    return VectorStoreIndexWrapper(vectorstore=vectorstore)


index = reload_index()

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# Function to save chat history
def save_chat_history(chat_history):
    with open("chat_history.txt", "a") as file:  # Open in append mode
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"\nConversation started at: {current_time}\n")
        for i, (user_query, response) in enumerate(chat_history):
            file.write(f"Q{i+1}: {user_query}\n")
            file.write(f"A{i+1}: {response}\n\n")

# Function to save specific conversations for training
def save_training_data(user_query, response):
    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/convo_training_data.txt", "a") as file:
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        file.write(f"Conversation at: {current_time}\n")
        file.write(f"Prompt: {user_query}\n")
        file.write(f"Response: {response}\n\n")
    # Reload index with updated data
    global index
    index = reload_index()

# Function to read existing chat history
def load_chat_history():
    if os.path.exists("chat_history.txt"):
        with open("chat_history.txt", "r") as file:
            history = file.read().strip().split("\n\n")
            loaded_history = []
            for entry in history:
                parts = entry.split("\n")
                if len(parts) == 2:
                    loaded_history.append((parts[0].split(": ", 1)[1], parts[1].split(": ", 1)[1]))
            return loaded_history
    return []

chat_history = load_chat_history()

while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    
    if query.lower() == 'ak47333':
        index = reload_index()
        print("Index reloaded successfully.")
        
       
    is_training_data = query.endswith("ak47")
    if is_training_data:
        query = query.replace("ak47", "").strip()
    
    result = chain.invoke({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    save_chat_history([(query, result['answer'])])  # Save only the new query and response
    
    if is_training_data:
        save_training_data(query, result['answer'])
    
    query = None
