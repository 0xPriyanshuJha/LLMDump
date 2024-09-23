from phi.assistant import Assistant
from phi.tools.serpapi_tools import SerpApiTools
from phi.llm.ollama import Ollama
import streamlit as st
from textwrap import dedent
import os
import dotenv


dotenv.load_dotenv()

SERPAPI = os.getenv("SERPAPI")


st.title("Llama2 powered Personal Finance Planner")
st.caption("Manage your finances with AI Personal Finance Manager by creating personalized budgets, investment plans, and savings strategies using Meta Llama2")



researcher = Assistant(
    name="Researcher",
    role="Searches for financial advice, investment opportunities, and savings strategies based on user preferences",
    llm=Ollama(model="llama2"),
    description=dedent(
        """\
        You are a world-class financial researcher. Given a user's financial goals and current financial situation,
        generate a list of search terms for finding relevant financial advice, investment opportunities, and savings strategies.
        Then search the web for each term, analyze the results, and return the 10 most relevant results.
        """
    ),
    instructions=[
        "Given a user's financial goals and current financial situation, first generate a list of 3 search terms related to those goals.",
            "For each search term, `search_google` and analyze the results.",
            "From the results of all searches, return the 10 most relevant results to the user's preferences.",
            "Remember: the quality of the results is important.",
    ],
    tools=[SerpApiTools(api_key=SERPAPI)],
    add_datetime_to_instructions=True,
    )
planner = Assistant(
        name="Planner",
        role="Generates a personalized financial plan based on user preferences and research results",
        llm=Ollama(model="llama2"),
        description=dedent(
        """\
        You are a senior financial planner. Given a user's financial goals, current financial situation, and a list of research results,
        your goal is to generate a personalized financial plan that meets the user's needs and preferences.
        """
        ),
    instructions=[
            "Given a user's financial goals, current financial situation, and a list of research results, generate a personalized financial plan that includes suggested budgets, investment plans, and savings strategies.",
            "Ensure the plan is well-structured, informative, and engaging.",
            "Ensure you provide a nuanced and balanced plan, quoting facts where possible.",
            "Remember: the quality of the plan is important.",
            "Focus on clarity, coherence, and overall quality.",
            "Never make up facts or plagiarize. Always provide proper attribution.",
        ],
        add_datetime_to_instructions=True,
        add_chat_history_to_prompt=True,
        num_history_messages=3,
    )

financial_goals = st.text_input("What are your financial goals?")
current_situation = st.text_area("Describe your current financial situation")

if st.button("Generate Financial Plan"):
    with st.spinner("Processing..."):
        response = planner.run(f"Financial goals: {financial_goals}, Current situation: {current_situation}", stream=False)
        st.write(response)