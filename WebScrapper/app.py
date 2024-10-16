import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph

st.title("Your Own Web Scrapper")
st.caption("Scrape the web for any information you need using SmartScraperGraph AI")


graph_config ={
    "llm":{
        "model": "ollama/llama2",
        "temperature": 0,
        "format": "json",
        "base_url": "http://localhost:11434"
    },
    "embeddings":{
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434"
    },
    "verbose": True
}

url = st.text_input("Enter the URL you want to scrape")
user_prompt = st.text_input("What you want the AI Agent to scrape for you?")

smart_scraper_graph = SmartScraperGraph(
    prompt = user_prompt,
    source = url,
    config = graph_config
)

if st.button("Scrape"):
    res = smart_scraper_graph.run()
    st.write(res)