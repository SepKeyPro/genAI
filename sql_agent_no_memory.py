import psycopg2
from langchain.llms import OpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.memory import ConversationBufferMemory
import os
from getpass import getpass

os.environ['OPENAI_API_KEY'] = getpass('Enter your API Key: ')


# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:Mahsa3504@localhost:5432/Office",
)

API_KEY = os.getenv('OPENAI_API_KEY')
llm = OpenAI(temperature=0, openai_api_key=API_KEY)

# Create db chain
QUERY = """
You are a home agent. Given an input question, first create a syntactically correct postgresql query to run, then look at the results of the query and return the answer.\n\n

{question}

Answer:
"""


# Setup the database chain
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)

def chat():
    print("Chatbot: Hi! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        try:
            question = QUERY.format(question=user_input)
            response = db_chain.run(question)
            print(f"Chatbot: {response}")
        except Exception as e:
            print(e)
    
chat()