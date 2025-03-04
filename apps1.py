# 1. Configure Langsmith
# 2. Import LLM
# 3. Import Data
# 4. Dynamic few-shot prompt
# 5. Custom SQL Tools
# 6. ReAct Agent Executor
# 7. Persistent Memory
# 8. Showcase in Gradio UI

# Langsmith Configuration

import os
from dotenv import load_dotenv
# Replace this with your actual LangChain API key
load_dotenv()
langsmith_api_key = os.getenv("langsmith_api_key")

if langsmith_api_key is None:
    raise ValueError("LangChain API key is not set.")

# Set the API key in the environment
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key


import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Local SQL Agent"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key


# LLM


from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.1:8b-instruct-q4_0")  #"llama3.1"

# Database
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///D:/pracDB/chinook.db", sample_rows_in_table_info = 3)

print(db.table_info)

# Few Shot Examples
examples = [
    {   "input": "List all artists.", 
        "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of Albums.",
        "query": "SELECT COUNT(DISTINT(AlbumId)) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]
print(len(examples))

# Dynamic Example Selector
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2')
from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    embeddings,
    FAISS,
    k=2,
    input_keys=["input"],
    )

example_selector.vectorstore.search("How many arists are there?", search_type = "mmr")

# Prompt
system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct sqlite query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most 5 results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.

You have access to the following tools for interacting with the database:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of {tool_names}
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.
If you see you are repeating yourself, just provide final answer and exit.

Here are some examples of user inputs and their corresponding SQL queries:"""

from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate

dynamic_few_shot_prompt = FewShotPromptTemplate(
    example_selector = example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input"],
    prefix=system_prefix,
    suffix=""
)

from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=dynamic_few_shot_prompt),
        ("human", "{input}"),
        ("system", "{agent_scratchpad}"),
    ]
)


# Custom Tools
from langchain.cache import SQLiteCache
from langchain_community.tools.sql_database.tool import (
    QuerySQLDataBaseTool, 
    InfoSQLDatabaseTool, 
    ListSQLDatabaseTool, 
    QuerySQLCheckerTool,
)

# Define cache (required by some tools)
cache = SQLiteCache(database_path="cache.db")
from langchain.callbacks.base import Callbacks
from langchain_community.tools.sql_database.tool import QuerySQLCheckerTool

# Rebuild the tool to ensure proper initialization
QuerySQLCheckerTool.model_rebuild()

# Initialize tools
tools = [
    QuerySQLDataBaseTool(db=db),
    InfoSQLDatabaseTool(db=db),
    ListSQLDatabaseTool(db=db),
    QuerySQLCheckerTool(db=db, llm=llm)
]

# Debug: print tool description
print(QuerySQLDataBaseTool(db=db).description)

# Invoke the prompt
prompt_val = full_prompt.invoke(
    {
        "input": "How many artists are there?",
        "tool_names": [tool.name for tool in tools],
        "tools": [tool.name + " - " + tool.description.strip() for tool in tools],
        "agent_scratchpad": [],
    }
)

# Print the result
print(prompt_val.to_string())

# Agent Executor

from langchain.agents import AgentExecutor, create_react_agent
agent = create_react_agent(llm, tools, full_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

# History Management
last_k_messages = 4


from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id):
    chat_message_history = SQLChatMessageHistory(
    session_id=session_id, connection = "sqlite:///D:/pracDB/.db", table_name = "local_table"
    )

    messages = chat_message_history.get_messages()
    chat_message_history.clear()
    
    for message in messages[-last_k_messages:]:
        chat_message_history.add_message(message)
    
    print("chat_message_history ", chat_message_history)
    return chat_message_history


from langchain_core.runnables.history import RunnableWithMessageHistory

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)
# Gradio UI
import gradio as gr
import uuid
import sqlite3

# SQLite database file path (update this to your database file path)
DB_PATH = "D:/pracDB/chinook.db"

def query_database(query):
    """Executes a SQL query on the SQLite database and returns the result."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    return result
    

def respond(message, chatbot_history, session_id):
    if not chatbot_history:
        session_id = uuid.uuid4().hex

    print("Session ID: ", session_id)

    # Handle user queries dynamically
    #try:
        # Attempt to process the message as an SQL query
    result = query_database(message)
    if isinstance(result, list) and result:
            # Format the results into a readable string
        response = "\n".join([", ".join(map(str, row)) for row in result])
    elif isinstance(result, str) and "Database error" in result:
        response = result  # Error message from query_database
    else:
            response = "No results found for the query."
    #except Exception as e:
        #response = f"Unable to process your request: {str(e)}"

    chatbot_history.append((message, response))
    return "", chatbot_history, session_id

# Gradio UI
with gr.Blocks() as demo:
    state = gr.State("")
    chatbot = gr.Chatbot()
    msg = gr.Textbox(label="Enter your SQL query:")
    clear = gr.ClearButton([msg, chatbot])

    msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])

demo.launch()



