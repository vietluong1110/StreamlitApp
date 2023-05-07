import tempfile
import os
#App Import
import streamlit as st
#LLMs Import
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferMemory
#Langchain PDF Loader
from langchain.document_loaders import PyPDFLoader 

#OPENAI API
os.environ['OPENAI_API_KEY'] = "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"

#Set Page Config 
st.set_page_config(page_icon="📃", page_title="PDF Reader")
#Set Page Title
st.title('📃 PDF Reader')

#Create Upload PDF File Element
uploaded_file = st.file_uploader(label="Upload your PDF File 📃", type='pdf')
if uploaded_file:
    st.markdown("Successfully Uploaded!")

#Create a prompt once the file is uploaded    
prompt = st.text_input(label="Your Question", placeholder="Your Query here about the PDF file",
                        autocomplete="What is...")
# If prompt without file
if prompt and not uploaded_file:
    st.error("Upload PDF File.")

#If both prompt and uploaded_file available
if prompt and uploaded_file:
    #Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file.flush()
        #Direct temporary file to PyPDF
        loader = PyPDFLoader(tmp_file.name)
        document = loader.load_and_split()
        #Text Splitter 
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        documents = text_splitter.split_documents(document)
        #OpenAIEmbeddings for the Texts
        embeddings = OpenAIEmbeddings()
        docsearch = Chroma.from_documents(documents,embeddings)
        #Similarity Search
        # similarity_search_docs = docsearch.similarity_search(prompt)

        #Load QA Retriver Chain
        qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0),
                                         chain_type='stuff',
                                         retriever=docsearch.as_retriever(),
                                         return_source_documents=True)
        result = qa({'query':prompt})
        st.write(result['result'])
        with st.expander("See Sources"):
            st.write("DOC CONTENT 1: {content}".format(content=result['source_documents'][0].page_content))
            st.write("PAGE: {page}".format(page=result['source_documents'][0].metadata['page']))
            st.write("------------------------------------------------")
            st.write("DOC CONTENT 2: {content}".format(content=result['source_documents'][1].page_content))
            st.write("PAGE: {page}".format(page=result['source_documents'][1].metadata['page']))
            st.write("------------------------------------------------")
            st.write("DOC CONTENT 3: {content}".format(content=result['source_documents'][2].page_content))
            st.write("PAGE: {page}".format(page=result['source_documents'][2].metadata['page']))

        os.remove(tmp_file.name)
        