from examples import db
from prompt import full_prompt
from environment_setup import llm

from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool, InfoSQLDatabaseTool, ListSQLDatabaseTool, QuerySQLCheckerTool

tools = [QuerySQLDataBaseTool(db = db), 
         InfoSQLDatabaseTool(db = db), 
         ListSQLDatabaseTool(db = db), 
         #QuerySQLCheckerTool(db = db, llm = llm)
         ]
print(QuerySQLDataBaseTool(db = db).description)

prompt_val = full_prompt.invoke(
    {
        "input": "{input}",
        "tool_names" : [tool.name for tool in tools],
        "tools" : [tool.name + " - " + tool.description.strip() for tool in tools],
        "agent_scratchpad": [],
    }
)

print(prompt_val.to_string())

from langchain.agents import AgentExecutor, create_react_agent
agent = create_react_agent(llm, tools, full_prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, handle_parsing_errors=True)

last_k_messages = 4


from langchain_community.chat_message_histories import SQLChatMessageHistory

def get_session_history(session_id):
    chat_message_history = SQLChatMessageHistory(
    session_id=session_id, connection = "sqlite:///D:/sql_local_chatbot/.db", table_name = "local_table"
    )

    messages = chat_message_history.get_messages()
    chat_message_history.clear()
    
    for message in messages[-last_k_messages:]:
        chat_message_history.add_message(message)
    
    print("chat_message_history ", chat_message_history)
    return chat_message_history


from langchain_core.runnables.history import RunnableWithMessageHistory

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor, # type: ignore
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

import gradio as gr
import uuid


with gr.Blocks() as demo:
    
    state = gr.State("")
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])


    def respond(message, chatbot_history, session_id):
        if not chatbot_history:
            session_id = uuid.uuid4().hex

        print("Session ID: ", session_id)

        response = agent_with_chat_history.invoke(
                                        {"input": message},
                                        {"configurable": {"session_id": session_id}},
                                        )

        chatbot_history.append((message, response['output']))
        return "", chatbot_history, session_id

    msg.submit(respond, [msg, chatbot, state], [msg, chatbot, state])

demo.launch()
