import streamlit as st
from phi.agent import Agent
from phi.tools.duckduckgo import DuckDuckGo
from phi.model.anthropic import Claude
from phi.tools.newspaper4k import Newspaper4k

# Initialize Claude model with the API key
anthropic_api_key = st.sidebar.text_input("Enter the API Key", type="password")

# Ensure the API key is not empty
if not anthropic_api_key:
    st.error("API Key is required")
else:
    anthropic_model = Claude(api_key=anthropic_api_key)

    # Proceed with the rest of your code
    search_tool = DuckDuckGo(search=True, news=True, fixed_max_results=5)

    # News Collector Agent
    news_collector = Agent(
        name="News Collector",
        role="Collects recent news articles on the given topic",
        tools=[search_tool],
        model=anthropic_model,
        instructions=["Gather latest articles on the topic"]
    )

    # Summary Writer Agent
    news_tool = Newspaper4k(read_article=True, include_summary=True)
    summary_writer = Agent(
        name="Summary Writer",
        role="Summarizes collected news articles",
        tools=[news_tool],
        model=anthropic_model,
        instructions=["Provide concise summaries of the articles"]
    )

    # Trend Analyzer Agent
    trend_analyzer = Agent(
        name="Trend Analyzer",
        role="Analyzes trends from summaries",
        model=anthropic_model,
        instructions=["Identify emerging trends and startup opportunities"]
    )

    # Agent Team
    agent_team = Agent(
        agents=[news_collector, summary_writer, trend_analyzer],
        instructions=[
            "Search news articles",
            "Summarize content",
            "Analyze trends",
            "Identify opportunities"
        ]
    )

    # Streamlit UI
    st.title("AI Startup Trend Analysis Agent")
    topic = st.text_input("Enter the area of interest:")

    news_response = news_collector.run(f"Collect recent news on {topic}")
    summary_response = summary_writer.run(f"Summarize: {news_response.content}")
    trend_response = trend_analyzer.run(f"Analyze trends: {summary_response.content}")
    st.write(trend_response.content)
