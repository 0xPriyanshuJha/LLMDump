import streamlit as st
from crewai import Agent, Crew, Task, Process
from langchain_ollama.llms import OllamaLLM
import ollama

OllamaLLM = None

def create_article(topic):
    research = Agent(
        role='Researcher',
        goal='Conduct thorough research on the given topic',
        backstory='You are an expert researcher with a keen eye for detail',
        verbose=True,
        allow_delegation=False,
        llm = OllamaLLM()
    )

    writer = Agent(
        role='Writer',
        goal = 'Write a well-structured article on the given topic',
        backstory = 'You are a skilled writer with a knack for storytelling',
        verbose=True,
        allow_delegation=False,
        llm = OllamaLLM()
    )

    editor = Agent(
        role = 'Editor',
        goal = 'Edit the article to ensure it is free of errors and reads well',
        backstory = 'You are an experienced editor with an eye for detail',
        verbose = True,
        allow_delegation = False,
        llm = OllamaLLM()
    )

    research_task = Task(
        description=f"Conduct comprehensive research on the topic: {topic}. Gather key information, statistics, and expert opinions.",
        agent=research,
        expected_output="A comprehensive research report on the given topic, including key information, statistics, and expert opinions."
    )

    write_task = Task(
        description="""Using the research provided, write a detailed and engaging article. 
        Ensure proper structure, flow, and clarity. Format the article using markdown, including:
        1. A main title (H1)
        2. Section headings (H2)
        3. Subsection headings where appropriate (H3)
        4. Bullet points or numbered lists where relevant
        5. Emphasis on key points using bold or italic text
        Make sure the content is well-organized and easy to read.""",
        expected_output="A well-written article on the given topic, structured in a clear and engaging manner."
    )

    edit_task = Task(
        description="""Review the article for clarity, accuracy, engagement, and proper markdown formatting. 
        Ensure that:
        1. The markdown formatting is correct and consistent
        2. Headings and subheadings are used appropriately
        3. The content flow is logical and engaging
        4. Key points are emphasized correctly
        Make necessary edits and improvements to both content and formatting.""",
        agent=editor,
        expected_output="A polished article on the given topic, free of errors and well-structured."
    )

    article_process = Process(
        name='Article Creation Process',
        tasks=[research_task, write_task, edit_task]
    )

    # Execute the process
    crew = Crew(process=article_process)
    crew.execute()

    return crew


# Page configuration
st.set_page_config(page_title="Multi Agent AI Researcher", page_icon="📝")

# Title
st.title("📝 Multi Agent AI Researcher")

# Sidebar for configuration (No API Key needed)
with st.sidebar:
    st.header("Configuration")
    st.info("Using Ollama open-source model. No API key required!")

# Main content
st.markdown("Generate detailed articles on any topic using AI agents!")

# Input for article topic
topic = st.text_input("Enter the topic for the article:", placeholder="e.g., The Impact of Artificial Intelligence on Healthcare")

if st.button("Generate Article"):
    if not topic:
        st.warning("Please enter a topic for the article.")
    else:
        with st.spinner("🤖 AI agents are working on your article..."):
            model = "llama2"
            prompt = f"Write a detailed article on the following topic: {topic}"
            response = ollama.chat(model=model, prompt=prompt)

            # Display the result in the app
            result = response['text']  # Assuming Ollama returns a 'text' field in the response
            st.markdown(result)

# Footer
st.markdown("---")
st.markdown("Powered by Ollama and Streamlit :heart:")