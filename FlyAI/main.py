import streamlit as st
import datetime
from crewai import Crew, LLM, Task, Agent
from crewai.tools import tool
from typing import Optional
from playwright.sync_api import sync_playwright
from html2text import html2text
from time import sleep
import os

st.set_page_config(page_title="Flight Search AI", page_icon="✈️")

# Initialize LLM
llm = LLM(
    model="ollama/deepseek-r1",
    url="https://localhost:11434"
)

@tool("Kayak tool")
def kayak_search(departure: str, destination: str, date: str, return_date: Optional[str] = None) -> str:
    """
    Generates a Kayak URL for flights between departure and destination on the specified date.
    """
    URL = f"https://www.kayak.com/flights/{departure}-{destination}/{date}"
    if return_date:
        URL += f"/{return_date}"
    URL += "?currency=INR"
    return URL

kayak = kayak_search

@tool("Browserbase tool")
def browserbase(url: str):
    """Loads a URL using a headless web browser and extracts text content."""
    with sync_playwright() as playwright:
        browser = playwright.chromium.connect_over_cdp(
            "wss://connect.browserbase.com?apiKey=" + os.environ["BROWSERBASE_API_KEY"]
        )
        context = browser.contexts[0]
        page = context.pages[0]
        page.goto(url)
        sleep(25)
        content = html2text(page.content())
        browser.close()
        return content

flight_agent = Agent(
    role="Flight Search Agent",
    goal="Find and Compare flights based on user preferences.",
    backstory="I specialize in searching for flights, providing options based on price, duration, and availability.",
    tool=[kayak, browserbase],
    allow_delegation=False,
)

search_task = Task(
    description="Search flight according to criteria {request}. Current Year:{current_year}",
    expected_output="""
    Here are our top 5 flights from Delhi to Bangalore on 4th March 2025:
    1. Indigo Airlines: Departure: 21:30, Arrival: 23:30, Duration: 2 Hours, Price: 5000 INR
    """,
    agent=flight_agent,
)

summarize_agent = Agent(
    role="Summarize Agent",
    goal="Summarize text and the results of the flight search while preserving key details and clarity",
    backstory="I specialize in summarizing content efficiently, providing a concise overview of the information.",
    allow_delegation=False,
)

summarize_task = Task(
    description="Summarize the raw search result",
    expected_output="""
    Here are our top 5 flights from Delhi to Bangalore on 4th March 2025:
    1. Indigo Airlines: Departure: 21:30, Arrival: 23:30, Duration: 2 Hours, Price: 5000 INR
    """,
    agent=flight_agent,
)

crew = Crew(
    agents=[flight_agent, summarize_agent],
    tasks=[search_task, summarize_task],
    max_rpm=100,
    verbose=True,
    planning=True,
)

st.title("✈️ AI-Powered Flight Search")
st.write("Enter your flight details and get the best available options!")

departure = st.text_input("Departure City (IATA Code)", "DEL")
destination = st.text_input("Destination City (IATA Code)", "BLR")
date = st.date_input("Departure Date", min_value=datetime.date.today())
return_date = st.date_input("Return Date (Optional)", value=None, min_value=datetime.date.today())

if st.button("Search Flights"):
    request = f"Flights from {departure} to {destination} on {date}" + (f" returning on {return_date}" if return_date else "")
    st.write("Searching flights...")
    res = crew.kickoff(
        inputs={
            "request": request,
            "current_year": datetime.date.today().year,
        }
    )
    st.subheader("Flight Search Results")
    st.write(res)
