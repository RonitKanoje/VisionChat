from typing import Annotated, TypedDict
import psycopg
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from dotenv import load_dotenv
import os
from langsmith import traceable
from .prompt import prompt1
    
load_dotenv()

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatOllama(model="llama3.2")

duck_tool = DuckDuckGoSearchRun()

wiki_tool = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(
        top_k_results=1,
        doc_content_chars_max=3000
    )
)

tools = [duck_tool, wiki_tool]
llm_with_tools = llm.bind_tools(tools,tool_choice='auto')


tool_node = ToolNode(tools)
@traceable(name = 'chat_node')
def chat_node(state: ChatState):
    messages = state["messages"]

    response = llm_with_tools.invoke(messages)

    return {"messages": [response]}

tool_node = ToolNode(tools)


conn = psycopg.connect(
    os.getenv("DATABASE_URL"), autocommit=True
)
checkpointer = PostgresSaver(conn=conn)

checkpointer.setup()

#Defining Graph
graph = StateGraph(ChatState)
graph.add_node('chat_node',chat_node)
graph.add_node("tools", tool_node)


graph.add_edge(START,'chat_node') 
graph.add_conditional_edges("chat_node",tools_condition)
graph.add_edge('tools','chat_node')

chatBot = graph.compile(checkpointer=checkpointer)

if __name__ == "__main__":      
    config = {"configurable": {"thread_id": "596559"}}

    initial_state = {
    "messages": [
        SystemMessage(content=prompt1),
        HumanMessage(content="Hello my name is Ronit")
        ]
    }

    response = chatBot.invoke(initial_state, config=config)

    # for message_chunk, metadata in chatBot.stream(
    #     {"messages": [HumanMessage(content="What is the capital of Gujarat?")]},
    #     config=config,
    #     stream_mode="messages"):
    #     print(message_chunk)

    # response = chatBot.invoke({'messages': [HumanMessage(content='Hello My name is Ronit')]}, config=config)

    print(response['messages'][-1].content)
    



def retrieveThreads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(
            checkpoint.config["configurable"]["thread_id"]
        )
    return list(all_threads)