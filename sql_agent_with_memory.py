from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.sql_database import SQLDatabase

import os
from getpass import getpass

os.environ['OPENAI_API_KEY'] = getpass('Enter your API Key: ')

API_KEY = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(api_key=API_KEY)

# Setup database
db = SQLDatabase.from_uri(
    f"postgresql+psycopg2://postgres:Mahsa3504@localhost:5432/Office",
)

def get_schema(_):
    return db.get_table_info()

# Prompt
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

template = """Given an input question, convert it to a SQL query. No pre-amble. Based on the table schema below, write a SQL query that would answer the user's question:
{schema}
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory
from langchain_core.runnables import RunnableLambda

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save


# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)

full_chain = (
    RunnablePassthrough.assign(query=sql_response_memory)
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)



def chat():
    print("Chatbot: Hi! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Chatbot: Goodbye!")
            break
        try:
            response = full_chain.invoke({"question": user_input})
            print(f"Chatbot: {response}")
        except Exception as e:
            print(e)
    
chat()