import streamlit as st 
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

import os
from dotenv import load_dotenv
load_dotenv()

## langsmith Tracking

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_TRACKING_V2"] = 'true'
os.environ['LANGCHAIN_PROJECT']= "Q&A ChaiBot with Ollama"

## prompt Template
prompt  = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant. Please response to the user queries"),
        ("user", "Question :{question}")
    ]
)

def generate_response(question, llm, temperature, max_tokens):
    llm = ChatGroq(model = llm, groq_api_key= "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")
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer


## title of the app
st.title('Enhanced Q&a cahtbot with ollama')

st.sidebar.title("Setting")

llm =  st.sidebar.selectbox("Select a model",["gemma2-9b-it","llama-3.1-70b-versatile", 'llama-3.2-90b-vision-preview'])

temperature = st.sidebar.slider("Temperature", min_value =0.0, max_value = 1.0, value = 0.7)
max_tokens = st.sidebar.slider("Max Tokens", min_value = 50, max_value = 300, value = 150)

st.write("Go ahead and ask any question")
user_input = st.text_input("you:")

if user_input:
    response = generate_response(user_input, llm , temperature, max_tokens)
    st.write(response)
else:
    st.write("please query")