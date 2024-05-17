import os
from crewai import Agent, Crew, Task, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools.graphql.tool import BaseGraphQLTool
from gql import Client, gql
from gql.transport.requests import RequestsHTTPTransport

# Set the API keys
os.environ["OPENAI_API_KEY"] = "your_openai_api_key"  # Replace with your actual OpenAI API key

# Initialize the LLM
llm = ChatOpenAI(
    model="llama3:8b",
    base_url="http://localhost:11434/v1"
)

# Define the GraphQL client
transport = RequestsHTTPTransport(
    url='https://api.spacex.land/graphql/',
    verify=True,
    retries=3,
)
client = Client(transport=transport, fetch_schema_from_transport=True)

# Define the GraphQL wrapper as a dictionary
graphql_wrapper = {
    "graphql_endpoint": 'https://api.spacex.land/graphql/'
}

# Initialize the GraphQL tool
graphql_tool = BaseGraphQLTool(graphql_wrapper=graphql_wrapper)

# Define the GraphQL Query Agent
graphql_query_agent = Agent(
    role='GraphQL Query Handler',
    goal='Execute GraphQL queries and return results',
    verbose=True,
    memory=True,
    backstory='An expert in handling GraphQL queries.',
    tools=[graphql_tool],
    allow_delegation=True,
    llm=llm
)

# Define the User Interface Agent
user_interface_agent = Agent(
    role='User Interface',
    goal='Interact with the user and gather input',
    verbose=True,
    memory=True,
    backstory='A friendly interface to gather user input.',
    tools=[],
    allow_delegation=False,
    llm=llm
)

# Define the task for querying GraphQL
query_task = Task(
    description='Execute a GraphQL query',
    agent=graphql_query_agent,  # Corrected attribute name
    expected_output='should receive a well-formed GraphQL response',
    required_tools=['graphql_tool']
)

# Define the task for user interaction
user_task = Task(
    description='Gather user input for GraphQL query',
    agent=user_interface_agent,  # Corrected attribute name
    expected_output='Interpret the GraphQL response into natural language'
)

# Create the crew
crew = Crew(
    agents=[graphql_query_agent, user_interface_agent],
    tasks=[user_task, query_task],
    process=Process.sequential,  # Execute tasks sequentially
    memory=True,
    cache=True,
    max_rpm=5,
    share_crew=True
)

# Execute the crew's process
if __name__ == "__main__":
    result = crew.kickoff(inputs={'topic': 'AI in healthcare'})
    print(result)