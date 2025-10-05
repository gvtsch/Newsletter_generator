import os
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

print("OpenAI API Key loaded successfully." if OPENAI_API_KEY else "Failed to load API Key.")
print("Gemini API Key loaded successfully." if GEMINI_API_KEY else "Failed to load Gemini API Key.")
print("SERPER API Key loaded successfully." if SERPER_API_KEY else "Failed to load SERPER API Key.")

import streamlit as st
import json
import requests

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper # This class acts as a wrapper, making it easier to send search queries to the Serper API and process the results in Python.


def search_serp(query):    
    
    search = GoogleSerperAPIWrapper(
        k=5,
        serper_api_key=SERPER_API_KEY
    )
    
    return search.results(query)


def pick_best_articles_urls(response_json, query):
    
    response_str = json.dumps(response_json)

    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
    )

    template = """
        You are an expert at picking the most relevant articles for a given query.
        You are a world-class researcher and have access to a vast amount of knowledge.
        You are able to quickly and accurately find the most relevant articles for any query.
        You are able to read and understand the content of articles and determine their relevance to the query.
        You are able to provide a list of the most relevant articles for the query.
        
        Query: {query}
        Response: {response}

        
        Please choose the best 3 articles from the list and return ONLY an array of the urls.  
        Do not include anything else -
        return ONLY an array of the urls. 
        Also make sure the articles are recent and not too old.
        If the file, or URL is invalid, show www.google.com.
        """

    prompt = PromptTemplate(
        input_variables=["query", "response"],
        template=template,
        verbose=False,
    )

    chain = prompt | llm
    
    urls = chain.invoke(
        {
            "query": query,
            "response": response_str,
        }
    )

    url_list = json.loads(urls.content)
    return url_list


def get_article_content(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(data)

    db_faiss = FAISS.from_documents(
        docs, 
        OpenAIEmbeddings()
    )

    return db_faiss


def summarize_content(db_faiss, query, k=3):
    docs = db_faiss.similarity_search( query, k=k )
    docs_page_content = " ".join([doc.page_content for doc in docs])
    
    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
    )

    template = """
        {docs}
        
        As an expert journalist and newsletter writer, create a compelling newsletter section about {query} based on the content above.
        
        Format: Write in the style of Tim Ferriss' "5-Bullet Friday" newsletter - engaging, informative, and actionable.
        
        Guidelines:
        1/ ENGAGING CONTENT: Hook readers with compelling insights, surprising facts, or thought-provoking questions
        2/ OPTIMAL LENGTH: Keep it concise - ideal for a single newsletter bullet point (150-300 words)
        3/ TOPIC FOCUS: Directly address {query} with relevant, up-to-date information
        4/ DATA-DRIVEN: Include specific examples, statistics, or case studies when available
        5/ READABILITY: Use clear, conversational language with short paragraphs and bullet points
        6/ ACTIONABLE VALUE: Provide concrete next steps, tools, resources, or links readers can immediately use
        7/ STRUCTURE: Start with a hook, present key insights, and end with actionable takeaways
        
        Output a newsletter section that readers will want to share and act upon.
        
        NEWSLETTER SECTION:
    """

    prompt = PromptTemplate(
        input_variables=["docs", "query"],
        template=template,
        verbose=False,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "docs": docs_page_content,
            "query": query,
        }
    )

    return response.content


def generate_newsletter(summaries, query):
    summaries_str = str(summaries)

    llm = ChatOpenAI(
        temperature=0.7,
        model="gpt-3.5-turbo",
    )

    template = """
        {summaries_str}
        
        As a world-class journalist and newsletter writer, create an engaging weekly newsletter about {query} using the content above.
        
        FORMAT: Tim Ferriss' "5-Bullet Friday" style - informal, personal, and action-oriented.
        
        OPENING: Start with:
        "Hi All!
        Here is your weekly dose of the Tech Newsletter, a list of what I find interesting and worth exploring."
        
        STRUCTURE:
        1/ PERSONAL BACKSTORY: Write 2-3 sentences sharing a personal, engaging, and lighthearted story or observation about {query}
        2/ MAIN CONTENT: Present the key insights in an easy-to-digest format
        3/ ACTIONABLE SECTION: Provide specific next steps, tools, or resources readers can use immediately
        
        GUIDELINES:
        • Keep content engaging and informative with concrete data
        • Write conversationally - no formal salutations
        • Make it scannable with bullet points and short paragraphs
        • Include Amazon links for books/products (or placeholders like [Amazon Link])
        • Focus specifically on {query} throughout
        • Provide actionable advice and insights
        

        
        
        "Christoph"
        
        NEWSLETTER:
    """

    prompt = PromptTemplate(
        input_variables=["summaries_str", "query"],
        template=template,
        verbose=False,
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "summaries_str": summaries_str,
            "query": query,
        }
    )

    return response.content