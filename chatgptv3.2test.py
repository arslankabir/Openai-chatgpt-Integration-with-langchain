import os,time
import sys
import openai
from datetime import datetime
import shutil
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

PERSIST_DIR = "persist"
CONVO_DATA_PATH = "data/convo_training_data.txt"
LAST_INDEX_TIME_FILE = "last_index_time.txt"

# Initialize OpenAI embeddings
embedding = OpenAIEmbeddings()  # Ensure you have access to OpenAI API and it's configured

# Remove the persisted directory if it exists
# def delete_persisted_data():
#     """Delete the persisted index data."""
#     if os.path.exists(PERSIST_DIR):
#         shutil.rmtree(PERSIST_DIR)
#         print(f"Deleted persisted directory: {PERSIST_DIR}")

# if os.path.exists(PERSIST_DIR):
#     shutil.rmtree(PERSIST_DIR)
#     print(f"Deleted persisted directory: {PERSIST_DIR}")


def get_last_modification_time(file_path):
    """Return the last modification time of a file."""
    if os.path.exists(file_path):
        return os.path.getmtime(file_path)
    return 0

def get_last_index_time():
    """Return the last time the index was created."""
    if os.path.exists(LAST_INDEX_TIME_FILE):
        with open(LAST_INDEX_TIME_FILE, "r") as f:
            return float(f.read().strip())
    return 0

def set_last_index_time(time_value):
    """Set the last index time."""
    with open(LAST_INDEX_TIME_FILE, "w") as f:
        f.write(str(time_value))

def is_data_updated(file_path, last_index_time):
    """Check if the data file has been updated since the last index."""
    last_modification_time = get_last_modification_time(file_path)
    return last_modification_time > last_index_time

def reload_index():
    last_index_time = get_last_index_time()
    
    if is_data_updated(CONVO_DATA_PATH, last_index_time):
        
        print("Creating new index due to updated data...\n")
        try:
            loader = DirectoryLoader("data/")
            text_loader = TextLoader(CONVO_DATA_PATH)
            
            # Debug: Check loaded data
            print("Loading data from directory and text file...")
            dir_data = loader.load()
            txt_data = text_loader.load()
            print(f"Directory Data: {len(dir_data)} documents loaded.")
            for i, doc in enumerate(dir_data):
                print(f"Document {i+1}: {doc.page_content[:100]}...")
            print(f"Text File Data: {len(txt_data)} documents loaded.")
            for i, doc in enumerate(txt_data):
                print(f"Document {i+1}: {doc.page_content[:100]}...")
            
            vectorstore_kwargs = {"persist_directory": PERSIST_DIR} if PERSIST else {}
            index_creator = VectorstoreIndexCreator(vectorstore_cls=Chroma, vectorstore_kwargs=vectorstore_kwargs, embedding=embedding)
            index = index_creator.from_loaders([loader, text_loader])
            vectorstore = index.vectorstore
            print("Index created successfully.")
            
            # Update the last index time
            set_last_index_time(time.time())
        except Exception as e:
            print(f"Error creating index: {e}")
            return None
    else:
        print("Reusing index from persist...\n")
        vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding)
    return VectorStoreIndexWrapper(vectorstore=vectorstore)

# Call the reload_index function to check if it runs successfully
index = reload_index()
if index is None:
    print("Failed to create or load the index.")
else:
    print("Index loaded successfully.")

# data_files = ["data/convo_training_data.txt", "data/data.txt"]  # List all relevant data files
# if is_data_updated(data_files):
#     index = reload_index()
#     print("Data updated, reloaded index.")
# else:
#     print("No new data, using existing index.")




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
        file.write(f"Instruction for future inquiries: {user_query}\n")
        print("Training is Updated!")
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

    if query.lower() == 'resetak47':
        delete_persisted_data()
        index = reload_index()
        print("Index reloaded after reset.")

        
       
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
