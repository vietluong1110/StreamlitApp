### Important API Keys
import os
os.environ["GOOGLE_CSE_ID"] = "b5b12ef5727a64d5e"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAf5FixWGn5uM6VFcP43d9CjVc_awK-Tws"
os.environ["OPENAI_API_KEY"] = "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"
#Langchain Imports
from langchain.document_loaders import SeleniumURLLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
#Other Imports
import streamlit as st
import time


#Set Page Config 
st.set_page_config(page_icon="🔗", page_title="URL Reader")
#Set Page Title
st.title('🔗 URL Reader')

## Set up the web elements

#URL and BUTTON
URL = st.text_input(label="Your URL", placeholder="Link")

#QUERY
query = st.text_input(label="Query:", placeholder="Your query about the URL...")
if query and not URL:
    st.error("Insert URL...")

#CHANNEL BUTTON
button = st.button(label="ASK")
if button and not URL:
    st.error("Please insert URL.")
if button and URL and not query:
    st.error("Please ask your questions")

if button and URL and query:
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.1)
        my_bar.progress(percent_complete + 1)

    loader = SeleniumURLLoader(urls=[URL])
    data = loader.load()    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    docs = db.similarity_search(query)
    chain = load_qa_chain(llm=ChatOpenAI(temperature=0), chain_type='stuff')
    answer = chain.run(input_documents=docs, question=query)

    st.write(answer)