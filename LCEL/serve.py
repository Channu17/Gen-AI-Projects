from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langserve import add_routes

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')
system_template = 'Translate the following into {language}'
model = ChatGroq(model= 'Gemma2-9b-It', groq_api_key= groq_api_key)
prompt_template = ChatPromptTemplate.from_messages([
    ('system', system_template),
    ('user', '{text}')
])

parser = StrOutputParser()

chain = prompt_template|model|parser


app = FastAPI(title='Langchain Server',
              version= '1.0',
              description ='A simple API server using Langchain runnable interfaces'
)

add_routes(
    app,
    chain,
    path = '/chain'
)

if __name__ =='__main__':
    import uvicorn
    uvicorn.run(app, host= '127.0.0.1', port=8000)