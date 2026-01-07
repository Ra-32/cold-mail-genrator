from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
import os
from cleaned_text import clean_text
import streamlit as st
from dotenv import load_dotenv


  
if os.path.exists(".env"):
    load_dotenv()
else:
    os.getenv("GROQ_API_KEY")==st.secrets["GROQ_API_KEY"]

groq_key = os.getenv("GROQ_API_KEY")

class Chains:
    def __init__(self):
        self.llm=ChatGroq(api_key=groq_key,model="llama-3.3-70b-versatile")

    def get_content(self, user_url):
        loader=WebBaseLoader(user_url)
        docs=loader.load()
        text = docs[0].page_content
        cleaned_text = clean_text(text)
        return cleaned_text
    

    def get_extract_job(self,user_url):
        cleaned_text=self.get_content(user_url)
        json_parser=JsonOutputParser()
        prompt=PromptTemplate(
        template="""
         ### SCRAPED TEXT FROM WEBSITE:
          {page_data}
          ### INSTRUCTION:
          The scraped text is from the career's page of a website.
          Your job is to extract the job postings and return them in JSON format containing the
          following keys: `role`, `experience`, `skills` and `description`.
          Only return the valid JSON.
          ### VALID JSON (NO PREAMBLE):
            {format_instruction}
          """,
        input_variables=["page_data"],
        partial_variables={"format_instruction":json_parser.get_format_instructions()}
        )

        chain= prompt | self.llm | json_parser
        response=chain.invoke({"page_data":cleaned_text})
        return response
    
