import os
import sys
import openai
from langchain.chains import ConversationalRetrievalChain
from langchain_openai.chat_models import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_chroma import Chroma  # Updated import

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY
 
# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True  # Set to True to persist the data

query = None
if len(sys.argv) > 1:
    query = sys.argv[1]

embedding = OpenAIEmbeddings()

if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=embedding)
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = DirectoryLoader("data/")
    vectorstore_kwargs = {"persist_directory": "persist"} if PERSIST else {}
    index = VectorstoreIndexCreator(vectorstore_cls=Chroma, vectorstore_kwargs=vectorstore_kwargs, embedding=embedding).from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(model="gpt-4"),
    retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

chat_history = []
while True:
    if not query:
        query = input("Prompt: ")
    if query in ['quit', 'q', 'exit']:
        sys.exit()
    result = chain.invoke({"question": query, "chat_history": chat_history})
    print(result['answer'])

    chat_history.append((query, result['answer']))
    query = None
