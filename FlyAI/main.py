from crewai import LLM, Task, Agent

llm = LLM(
    model = "ollama/deepseek-r1",
    url = "https://localhost:11434"
)

flight_agent = Agent(
    role = "Flight Search Agent",
    goal = "Find and Compare flights based on user preferences.",
    backstory = """I specialize in searching for flights, providing options based on price, duration and availability""",
    tool = [kayak, browserbase],
    allow_delegation = False,
)


search_task = Task(
    description = (
        "Search flight according to criteria {request}. Current Year:{current_year}"
    ),
    output_example = """
Here are our top 5 flights from Delhi to Bangalore on 4th March 2025:
1. Indigo Airlines: Departure: 21:30, Arrival: 23:30, Duration: 2 Hours, Price: 5000 INR
""",
agent = flight_agent,
)


summarize_agent = Agent(
    role = "Summarize Agent",
    goal = "Summarize text and the results of the flight search while preserving key details and clarity",
    backstory = """I specialize in summarizing content efficiently, providing a concise overview of the information""",
    allow_delegation = False,
)

summarize_task = Task(
    description = "Summarize the raw search result",
    expected_output = """Here are our top 5 flights from Delhi to Bangalore on 4th March 2025: 
    1. Indigo Airlines: 
        - Departure: 21:30, 
        - Arrival: 23:30, 
        - Duration: 2 Hours, 
        - Price: 5000 INR
        - Booking: [Indigo Airlines](https://kayak.com/)
        ...
        """,
    agent = flight_agent,
)

