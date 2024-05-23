import os
from gql import gql, Client
from gql.transport.aiohttp import AIOHTTPTransport

from crewai import Crew, Agent, Task, Process
from crewai_tools import tool
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "NA"  # Replace with your actual OpenAI API key
os.environ["OPENAI_MODEL_NAME"] = "llama3:8b"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

# Initialize the LLM
llm = ChatOpenAI(
    model="llama3:8b",
    base_url="http://localhost:11434/v1"
)

@tool("get_spacex_graphql_schema")
def my_simple_tool():
    """Useful to understand the schema of a graphql service"""
    # Select your transport with a defined url endpoint
    transport = AIOHTTPTransport(url="https://main--spacex-l4uc6p.apollographos.net/graphql")

    # Create a GraphQL client using the defined transport
    client = Client(transport=transport, fetch_schema_from_transport=True)

    # Provide a GraphQL introspection query
    introspection_query = gql("""
    {
      __schema {
        types {
          name
        }
      }
    }
    """)

    # Execute the query on the transport
    result = client.execute(introspection_query)
    print(result)
    return result

# Define your agents
graph_analyst_agent = Agent(
  role='GraphQL Analyst',
  goal='Conduct analysis on possible queries on the graphql service',
  backstory='an experienced graphql analyst with great skills at unlocking value from graphql services',
  tools=[my_simple_tool],
)

analyst = Agent(
  role='Data Analyst',
  goal='Analyze research findings',
  backstory='A meticulous analyst with a knack for uncovering patterns',
  tools=[my_simple_tool],
)

writer = Agent(
  role='GraphQL Query Writer',
  goal='Create a queryQL query that supports the users request',
  backstory='A skilled writer with a talent for crafting compelling narratives',
  tools=[my_simple_tool]
  ,
)

# Define the tasks in sequence
research_task = Task(
    expected_output="List possible entites to run queries on",
    agent=graph_analyst_agent)

analysis_task = Task(
    description='Analyze the data...', 
    expected_output="List possible queries against the possible entities",
    agent=analyst)

writing_task = Task(
    description='Compose the a graphql query pertinent to the user request',
    expected_output='a validated query that will run against the graphql service',
    agent=writer,
    human_input=True)

# Form the crew with a sequential process
crew = Crew(
  agents=[graph_analyst_agent, analyst, writer],
  tasks=[research_task, analysis_task, writing_task],
  process=Process.sequential,
  embedder={
        "provider": "huggingface",
        "config": {
            "model": "mixedbread-ai/mxbai-embed-large-v1",  # Example model from HuggingFace
        }}
)

# Execute tasks
result = crew.kickoff()
print(result)