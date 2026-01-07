import streamlit as st
from langchain_groq import ChatGroq
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings

from chains import groq_key, Chains
from portfolio import Portfolio


llm=ChatGroq(api_key=groq_key,model="llama-3.3-70b-versatile")
st.title("Cold Mail Genrator ...✅✉️")

st.header("Genrate a cold mail")
user_url=st.text_input("Enter the careers page URL")
chains=Chains()
portfolio=Portfolio()
if st.button("Generate") and user_url:
    
    json_response=chains.get_extract_job(user_url)
    links_list=portfolio.load_portfolio(user_url)
    email_content=portfolio.get_email(user_url,links_list=links_list)
    st.write(email_content)


   

    


