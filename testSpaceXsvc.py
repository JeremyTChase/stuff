# Warning control
import warnings
warnings.filterwarnings('ignore')

import os
from langchain_openai import ChatOpenAI
from crewai import Agent, Task, Crew, Process
from crewai_tools import ScrapeWebsiteTool, SerperDevTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["X-API-KEY"] = "dffe22e4715123b3910f3fb6bb05dbb57e2c154f"

# Initialize the LLM in Ollama
llm = ChatOpenAI(
    model="llama3:8b",
    base_url="http://localhost:11434/v1",
    temperature=0.5
)

# Define the support agent
GraphQL_Architect = Agent(
    role="GraphQL Architect",
    goal="Expert in designing conversational flows and integrating with GraphQL services",
    backstory=(
        "You are an architect who can provide a solution to interact with a GraphQL service "
        "You need to make sure that you provide the best support! "
        "Make sure to provide full complete answers, and make no assumptions."
    ),
    allow_delegation=True,
    Cache = False,
    verbose=True,
    memory=False, # Enable memory for the agent
    tools = [scrape_tool, search_tool]
)

# Define the support agent
graphql_agent = Agent(
    role="GraphQL Service Discovery Specialist",
    goal="Expert in dynamically discovering available services through GraphQL queries",
    backstory=(
        "You are an expert GraphQL engineer who can interact with GraphQL services using GraphQL "
        "You are able to discover services of the provided graph service {url} by intergrating the service using graphql"
        "Make a list of available queries for that specific GraphQL service in table form"
    ),
    allow_delegation=True,
    Cache = False,
    verbose=True,
    memory=False,  # Enable memory for the agent
    tools = [scrape_tool, search_tool]
)

# Task for Data Analyst Agent: Analyze Market Data
GraphQLServiceAnalysis = Task(
    description=(
        "Examine the graphQL service"
        "Use discovery techniques to discover the available queries for the GraphQL service"
        "identify trends and predict market movements."
    ),
    expected_output=(
        "Table of queries and their related input and outputs"
        "plus a brief descrption for the following service {url}."
    ),
    agent=GraphQL_Architect,
)

graphql_crew = Crew(
    agents=[GraphQL_Architect, 
            graphql_agent],
    
    tasks=[GraphQLServiceAnalysis],
    
    manager_llm=llm,
    process=Process.hierarchical,
    verbose=True,
    embedder={
        "provider": "huggingface",
        "config": {
            "model": "mixedbread-ai/mxbai-embed-large-v1",  # Example model from HuggingFace
          }
        },
    memory=False
)

inputs_obj = {
    "url": " http://docs.catalysis-hub.org/en/latest/tutorials/index.html#graphql"
}


result = graphql_crew.kickoff(inputs=inputs_obj)