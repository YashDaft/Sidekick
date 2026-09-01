from openai.types.responses import response
from langchain_core.messages import SystemMessage
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_tavily import TavilySearch
from langchain_core.tools import tool

import sqlite3
import requests
import math

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langgraph.types import interrupt, Command

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", google_api_key=api_key)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)



def ingest_rag_documents(file_path):
    DB_PATH = "faiss_db"
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(DB_PATH)


def get_retriever():
    DB_PATH = "faiss_db"
    vector_store = FAISS.load_local(
        folder_path=DB_PATH,
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    retriever = vector_store.as_retriever(
        search_type = 'similarity',
        search_kwargs = {"k": 4}
    )
    return retriever

@tool
def rag_tool(query: str) -> str:
    '''
    Retrieve relevant information from the PDF document.

    Use this tool when the user ask factual or conceptual questions 
    that may be answered using the stored pdf documents.

    Args:
      query: The question or search query used to retrieve PDF content.
    '''

    retriever = get_retriever()
    documents = retriever.invoke(query)

    if not documents:
        return 'No relevant information was found in the PDF.'
    
    formatted_documents = []

    for index, document in enumerate(documents, start=1):
        source = document.metadata.get('source', 'Unkown source')
        page = document.metadata.get('page', 'Unkown page')
        
        formatted_documents.append(
            f'Document {index}\n'
            f'Source: {source}\n'
            f'Page: {page}\n'
            f'Content: {document.page_content}'
        )
    return '\n\n'.join(formatted_documents)





search_tool = TavilySearch(
    max_results = 5,
    topic= "general",
    search_depth = 'advanced'
)

@tool 
def calculator(expression: str) -> str:
    '''useful for simple math calculations.
    Input should be a valid math expression.
    Example: 2 + 2, math.sqrt(16), 10*5'''

    try:
        allowed = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
            "sum": sum
        }

        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)

    except Exception as e:
        return f'Calculation error: {str(e)}'
         


@tool
def get_stock_price(symbol: str) -> str:
    '''
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alph Vintage with API key in the url.
    '''
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=6YK6ABAK36PXUCHT'
    r = requests.get(url)
    return r.json() 


@tool
def purchase_stock(symbol: str, quantity: str) -> dict:
    '''
    Simulate purchasing a stock of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt and
    wait for a human decision ("yes"/ anything else).
    '''
    decision = interrupt(f'Approve buying {quantity} shares of {symbol}? (yes / no)')

    if isinstance(decision, str) and decision.strip().lower() == 'yes':
        return {
            "status": "success",
            "message": f"purchase order placed for {quantity} shares of {symbol}.",
            "symbol": symbol,
            "quantity": quantity,
        }
    else:
        return {
            "status": "cancelled",
            "message": f"purchase of {quantity} shares of {symbol} was declined by human.",
            "symbol": symbol,
            "quantity": quantity
        }
    
    
        
 




OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")


@tool
def get_weather(latitude: float, longitude: float) -> dict:
    """
    Get the current weather for a given latitude and longitude.

    Args:
        latitude: Current location latitude.
        longitude: Current location longitude.

    Returns:
        A dictionary containing the current weather information.
    """

    if not OPENWEATHER_API_KEY:
        return {
            "error": "OPENWEATHER_API_KEY is not configured."
        }

    url = "https://api.openweathermap.org/data/2.5/weather"

    params = {
        "lat": latitude,
        "lon": longitude,
        "appid": OPENWEATHER_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()
        data = response.json()

        return {
            "location": data.get("name"),
            "country": data.get("sys", {}).get("country"),
            "temperature_c": data.get("main", {}).get("temp"),
            "feels_like_c": data.get("main", {}).get("feels_like"),
            "humidity_percent": data.get("main", {}).get("humidity"),
            "pressure_hpa": data.get("main", {}).get("pressure"),
            "weather": data.get("weather", [{}])[0].get("main"),
            "description": data.get("weather", [{}])[0].get("description"),
            "wind_speed_mps": data.get("wind", {}).get("speed"),
            "visibility_m": data.get("visibility"),
        }

    except requests.exceptions.Timeout:
        return {
            "error": "Weather API request timed out."
        }

    except requests.exceptions.RequestException as e:
        return {
            "error": f"Weather API request failed: {str(e)}"
        }

tools = [search_tool, get_stock_price, calculator, get_weather, rag_tool, purchase_stock]

llm_with_tools = llm.bind_tools(tools)


#state of the graph
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]




def chat_node(state: ChatState):
    '''LLM node that can answer directly or call an appropriate call.'''

    system_message = SystemMessage(
        content=(
            'You are a helpful Agentic Chatbot with access to several tools.\n\n'

            'Tool usage instructions:\n'
            '-Use `rag_tool` for questions about the uploaded PDF or document'
            '-Always retrieve relevant document content before answering PDF-related questions'
            '-Use `search_tool` for current events, recent information, or information'
            'that requires an internet search.\n'
            '-Use `calculator` for mathematical calculations. Do not calculate complex'
            'expressions manually when the calculator is available.\n'
            '-Use `get_stock_price` when the user asks for the current price of a stock.\n'
            'Use `get_current_weather` when the user asks about current weather for a location.\n\n'
            '-Use `purchase_stock` when the user asks to buy or purchase shares of a stock. '
            'This always requires human approval before the purchase is completed.\n'

            'Answer general questions directly when no tool is required'
            'Do not invent information from the uploaded document'
            'If the user asks about a PDF but no document is available, ask them to upload a PDF'
            'After recieving a tool result, provide a clear and helpful final answer'

            
        )
    )

    messages = [
        system_message,
        *state["messages"]
    ]

    response = llm_with_tools.invoke(messages)
    return {'messages': [response]}

    

tool_node = ToolNode(tools, handle_tool_errors=False)

connection = sqlite3.connect(database="agentic_chatbot.db", check_same_thread=False)
checkpoint = SqliteSaver(connection)
# Build Graph
graph = StateGraph(ChatState)

#add nodes
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)

#add edges
graph.add_edge(START, 'chat_node')
graph.add_conditional_edges('chat_node', tools_condition)
graph.add_edge('tools', 'chat_node')

# Compile Graph

chatbot = graph.compile(checkpointer=checkpoint)



#hlper function to get threads from db 
def get_all_threads():
    all_threads = set()
    for ckpt in checkpoint.list(None):
        all_threads.add(ckpt.config['configurable']['thread_id'])

    return list(all_threads)










