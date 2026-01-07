import pandas as pd
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from  chains import Chains, groq_key
from langchain_community.vectorstores import FAISS

import os
if os.path.exists(".env"):
    load_dotenv()
else:
    os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")==st.secrets["HUGGINGFACEHUB_ACCESS_TOKEN"]

hf_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
chains=Chains()

llm=ChatGroq(api_key=groq_key,model="llama-3.3-70b-versatile")

llm_embedding = HuggingFaceEndpointEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    task="feature-extraction",
    huggingfacehub_api_token=hf_token,
)

class Portfolio:
    def __init__(self, file_path="resources/my_portfolio.csv"):
        self.df=pd.read_csv(file_path)
        self.file_path=file_path
        

    # result is json data from get_extract_job

    def load_portfolio(self,user_url,file_path="resources/my_portfolio.csv"):
        results=chains.get_extract_job(user_url=user_url)
        docs = [
            Document(
                page_content=row["Techstack"],
                metadata={"links": row["Links"]}
            )
            for _, row in self.df.iterrows()
            ]
        vectorstore=FAISS.from_documents(

           docs,llm_embedding
        )
        vectorstore.save_local("faiss_portfolio")
    
        query = " ".join(results["skills"])
        
        results_doc = vectorstore.similarity_search(
            query=query,
            k=2
        )
        links = [doc.metadata["links"] for doc in results_doc]
        return links
       
    
    def get_email(self,user_url,links_list=None):
        results=chains.get_extract_job(user_url=user_url)
        links_list=self.load_portfolio(user_url=user_url)
        prompt_email = PromptTemplate.from_template(
        """
        ### JOB DESCRIPTION:
        {job_description}

        ### INSTRUCTION:
        You are Mohan, a business development executive at AtliQ. AtliQ is an AI & Software Consulting company dedicated to facilitating
        the seamless integration of business processes through automated tools.
        Over our experience, we have empowered numerous enterprises with tailored solutions, fostering scalability,
        process optimization, cost reduction, and heightened overall efficiency.
        Your job is to write a cold email to the client regarding the job mentioned above describing the capability of AtliQ
        in fulfilling their needs.
        Also add the most relevant ones from the following links to showcase Atliq's portfolio: {link_list}
        Remember you are Mohan, BDE at AtliQ.
        Do not provide a preamble.
        ### EMAIL (NO PREAMBLE):

        """
        )

        chain_email = prompt_email | llm
        res = chain_email.invoke({"job_description": str(results), "link_list": links_list})
        return res.content
        






