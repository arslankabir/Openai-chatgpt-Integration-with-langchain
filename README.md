OpenAI ChatGPT Integration with LangChain
Overview
This repository demonstrates the integration of OpenAI's ChatGPT (GPT-4) with LangChain to create a conversational AI system. The system provides an interactive platform where users can query data and receive intelligent responses based on a custom dataset. It supports data persistence, dynamic index management, and chat history tracking.

Features
Conversational AI: Leverages OpenAI's GPT-4 model to generate responses tailored to user queries.
Data Persistence: Stores vector store indices to optimize performance for repeated queries.
Dynamic Index Management: Automatically updates the index when data changes are detected, ensuring the most current information is used.
Chat History Tracking: Maintains a record of interactions to provide context in ongoing conversations.
Training Data Management: Allows saving specific interactions as training data to improve the model's performance.
Index Reset Functionality: Provides a mechanism to reset and rebuild the index if needed.
Installation
Prerequisites
Python 3.7+
An OpenAI API key
Required Python packages (specified in requirements.txt)
Setup
Clone the Repository:

sh
git clone https://github.com/arslankabir/chatgpt-Integration.git
cd chatgpt-Integration
Install Dependencies:

sh
pip install -r requirements.txt
Configure API Key:

Set your OpenAI API key in constants.py:

python
# constants.py
APIKEY = 'your_openai_api_key'
Usage
Running the System
To start the conversational AI system, run:

sh
python main.py
You can also start with a specific query:

sh
python main.py "Your query here"
Special Commands
ak47333: Reloads the index from persisted data.
resetak47: Deletes persisted data and recreates the index.
ak47: Marks a query to be added to the training data.
Chat History and Training Data
Chat History: Logged in chat_history.txt.
Training Data: Stored in data/convo_training_data.txt.
File Structure
bash
Copy code
chatgpt-Integration/
│
├── data/
│   ├── convo_training_data.txt  # Training data for the system
│   └── ...                      # Additional data files
│
├── persist/                     # Directory for persisted index data
│
├── last_index_time.txt          # Stores the timestamp of the last index creation
├── chat_history.txt             # Records of past conversations
│
├── constants.py                 # Configuration file for API keys and constants
├── main.py                      # Main application script
└── requirements.txt             # List of Python dependencies
Functions
reload_index()
Purpose: Checks if the data has been updated and reloads the vector store index if necessary. It uses the last modification time of the data files to determine if reindexing is needed.
save_chat_history(chat_history)
Purpose: Saves the chat history to a text file, appending each new interaction.
save_training_data(user_query, response)
Purpose: Saves specific conversations as training data to a designated file, helping to refine the AI's responses.
load_chat_history()
Purpose: Loads previous chat history from a file to maintain context in ongoing sessions.
Contributing
We welcome contributions to improve this project. Please follow these steps:

Fork the repository.
Create a new feature branch (git checkout -b feature-branch).
Make your changes and commit them (git commit -m 'Add new feature').
Push to your branch (git push origin feature-branch).
Open a Pull Request for review.
License
This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments
OpenAI for providing the GPT-4 model.
LangChain for the framework enabling conversational AI.
For more details and usage instructions, please refer to the documentation.
