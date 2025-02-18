from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.vectorbase.lancedb import lanceDb, SearchType
from agno.playground import Playground, serve_playground_app
from agno.tools.duckduckgo import DuckDuckGoTools

db_uri = "tmp/lanceDb"

knowledge_base = PDFUrlKnowledgeBase(
    urls = ["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db = lanceDb(table_name = "recipe", uri=db_uri, search_type = SearchType.vector),
)
knowledge_base.load(upsert=True)

rag_agent = Agent(
    model = OpenAIChat(id="gpt-4o"),
    agent_id = "rag_agent",
    knowledge = knowledge_base,
    tools = [DuckDuckGoTools()],
    show_tools_call = True,
    markdown = True,
)

app = Playground(agent=[rag_agent]).get_app()

if __name__ == "__main__":
    serve_playground_app("rag_agent:app", reload=True)