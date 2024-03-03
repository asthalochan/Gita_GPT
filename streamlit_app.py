import os
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone,ServerlessSpec
from langchain_community.llms import HuggingFaceHub
from langchain_community.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone as PineconeClient
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()
huggingfacehub_api_token=st.secrets["huggingfacehub_api_token"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_ENVIRONMENT = st.secrets["PINECONE_ENVIRONMENT"]
PINECONE_INDEX_NAME = st.secrets["PINECONE_INDEX_NAME"]


pinecone = PineconeClient(api_key=PINECONE_API_KEY,
                         environment=PINECONE_ENVIRONMENT)
index_name="gita-gpt"

#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
embeddings = download_hugging_face_embeddings()




vector_store = Pinecone.from_existing_index(index_name, embeddings)

st.session_state.vector_store=vector_store


prompt_template="""
Use the following pieces of information to answer the user's question with yours own easy word .
do not write in the context, from the context and point wise.
provide the complete sentences of answer in paragraph.
 
Context: {context}
Question: {question}

Answer:
"""
PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])


llm = HuggingFaceHub(repo_id='mistralai/Mistral-7B-Instruct-v0.2',model_kwargs={"temperature":0.1}, huggingfacehub_api_token=huggingfacehub_api_token)

qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
    return_source_documents=True, 
    chain_type_kwargs={"prompt": PROMPT})

def answer_format(text):
    start_index = text.find("Answer:")
    if start_index == -1:
        return None
    return text[start_index:].strip()


def get_response(user_query):


    result=qa({"query": user_query})
    result_text = result["result"]
    answer = answer_format(result_text)
    return answer


# app config
st.set_page_config(page_title="Chat with Bhagavad Gita", page_icon="🕉️")
st.title("Chat with Bhagavad Gita 🕉️")







# session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am a Bhagavad Gita bot. How can I help you?"),
    ]


# user input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))
    
    

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)



# footer
#st.write("Made with ❤️ by Asthalochan © 2024")
