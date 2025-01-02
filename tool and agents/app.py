import streamlit as st 
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings 
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler

import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key  =os.getenv("GROQ_API_KEY")
os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN")

arxiv_wrapper = ArxivAPIWrapper(top_k_results =1, doc_content_char_max = 200)
arxiv = ArxivQueryRun(api_wrapper = arxiv_wrapper)


wiki_wrapper = WikipediaAPIWrapper(top_k_wrapper = 1, doc_cotent_chars_max = 200)
wiki = WikipediaQueryRun(api_wrapper = wiki_wrapper)

search = DuckDuckGoSearchRun(name = 'search')

st.title("LangChain Chatbot")

if "messages" not in st.session_state:
    st.session_state['messages'] =[
        {'role':"assistant", 'content':"Hi! I'm a chatbot who can search the web. How can i help you"}
    ]
    
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

    
if prompt:=st.chat_input(placeholder = "what is machine learning?"):
    st.session_state.messages.append({'role':"user", 'content':prompt})
    st.chat_message("user").write(prompt)
    
    llm = ChatGroq(groq_api_key = groq_api_key, model_name = "llama3-8b-8192", streaming = True)    
    tools = [search, arxiv, wiki]
    search_agent = initialize_agent(tools, llm, agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION, handling_parsing_errors = True)   
    
    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts = False)
        try:
            response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
        except Exception as e:
            response = f"Sorry, an error occurred: {str(e)}"
        st.session_state.messages.append({'role': "assistant", 'content': response})
        st.write(response)