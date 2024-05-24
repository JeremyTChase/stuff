import os
import requests
from textwrap import dedent
from crewai import Crew, Agent, Task, Process 
from langchain_community.llms import Ollama
from crewai_tools import tool, PDFSearchTool

# Set your API keys
#os.environ["OPENAI_API_KEY"] = "sk-xxx"

llm = Ollama(
    #model="llama3:8b",
    model="mistral:latest",
    temperature=0.5
    )

spaceXSchema = PDFSearchTool(
    pdf='./spacex/spacexSchema.pdf',
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mistral:latest",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="mixedbread-ai/mxbai-embed-large-v1",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )    
)

spaceXExampleQuery = PDFSearchTool(
    pdf='./spacex/spacexExampleQueries.pdf',
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="mistral:latest",
                # temperature=0.5,
                # top_p=1,
                # stream=true,
            ),
        ),
        embedder=dict(
            provider="huggingface",
            config=dict(
                model="mixedbread-ai/mxbai-embed-large-v1",
                #task_type="retrieval_document",
                # title="Embeddings",
            ),
        ),
    )    
)

# spaceXSchema = PDFSearchTool(pdf='./spacex/spacexSchema.pdf')
# spaceXExampleQuery = pdfSearchTool(pdf='./spacex/spacexExampleQueries.pdf')

# Define the GraphQL endpoint
url = "https://main--spacex-l4uc6p.apollographos.net/graphql"


# Example: Loading from a file
# spaceXSchema = RagTool(    
#     config=dict(
#         llm=dict(
#             provider="ollama", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model="mistral:latest",
#                 # temperature=0.5,
#                 # top_p=1,
#                 # stream=true,
#             ),
#         ),
#         embedder=dict(
#             provider="huggingface",
#             config=dict(
#                 model="mixedbread-ai/mxbai-embed-large-v1",
#                 #task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
#     ).parse_file('./spaceXgraphQLSchema.gql')

# spaceXExampleQuery = RagTool(    
#     config=dict(
#         llm=dict(
#             provider="ollama", # or google, openai, anthropic, llama2, ...
#             config=dict(
#                 model="mistral:latest",
#                 # temperature=0.5,
#                 # top_p=1,
#                 # stream=true,
#             ),
#         ),
#         embedder=dict(
#             provider="huggingface",
#             config=dict(
#                 model="mixedbread-ai/mxbai-embed-large-v1",
#                 #task_type="retrieval_document",
#                 # title="Embeddings",
#             ),
#         ),
#     )
#     ).validate.model_validate(spaceXExampleQueryTxt)

@tool
def requestGraphqlQuery(query: str)->str:
    """useful for query the graphql service"""
    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Define the payload
    payload = {
        "query": query
    }

    # Make the request
    response = requests.post(url, json=payload, headers=headers)

    return response.json()


# @tool
# def GetGraphQLSchema()->str:
   
#     # Define the introspection query
#     introspection_query = """
#     {
#     __schema {
#         queryType {
#         name
#         }
#         mutationType {
#         name
#         }
#         subscriptionType {
#         name
#         }
#         types {
#         ...FullType
#         }
#         directives {
#         name
#         description
#         locations
#         args {
#             ...InputValue
#         }
#         }
#     }
#     }

#     fragment FullType on __Type {
#     kind
#     name
#     description
#     fields(includeDeprecated: true) {
#         name
#         description
#         args {
#         ...InputValue
#         }
#         type {
#         ...TypeRef
#         }
#         isDeprecated
#         deprecationReason
#     }
#     inputFields {
#         ...InputValue
#     }
#     interfaces {
#         ...TypeRef
#     }
#     enumValues(includeDeprecated: true) {
#         name
#         description
#         isDeprecated
#         deprecationReason
#     }
#     possibleTypes {
#         ...TypeRef
#     }
#     }

#     fragment InputValue on __InputValue {
#     name
#     description
#     type {
#         ...TypeRef
#     }
#     defaultValue
#     }

#     fragment TypeRef on __Type {
#     kind
#     name
#     ofType {
#         kind
#         name
#         ofType {
#         kind
#         name
#         ofType {
#             kind
#             name
#         }
#         }
#     }
#     }
#     """

#     # Define the headers
#     headers = {
#         "Content-Type": "application/json"
#     }

#     # Define the payload
#     payload = {
#         "query": introspection_query
#     }

#     # Make the request
#     response = requests.post(url, json=payload, headers=headers)

#     # Print the response
#     #print(response.json())
#     return response.json()

myAgent1 = Agent(
			role="Lead GraphQL Engineer for {service_title}",
			goal=dedent("""\
				develop plans based on user input that {user_input} 
                need graphql query support
               """),
			backstory=dedent("""\
				Lead GraphQL engineer with 20 years experience in developing GraphQL services
                    {service_description}
                    """),
			tools=[
					requestGraphqlQuery,
                    spaceXSchema,
                    spaceXExampleQuery
			],
            cache = True,
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

myAgent2 = Agent(
			role="Lead GraphQL Query Engineer for {service_title}",
			goal=dedent("""\
				develops a GraphQL queries to satisfy the user requst : {user_input} NOT python
               """),
			backstory=dedent("""\
				Lead GraphQl query engineer with 20 years experience in developing GraphQL queries
                    {service_description}
                    """),
			tools=[
					requestGraphqlQuery,
                    spaceXSchema,
                    spaceXExampleQuery
			],
            cache = True,
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

myAgent3 = Agent(
			role="Concierge for the {service_title} GraphQL service",
			goal=dedent("""\
				Works with the user to answer there query: {user_input}
               """),
			backstory=dedent("""\
				Helpful assistant to facilitate user needs on the graphql service and return answers
                    {service_description}
                """),
			tools=[
					requestGraphqlQuery,
                    spaceXSchema,
                    spaceXExampleQuery
			],
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

# Task examine the graphql service (gets schema)
task1 = Task(
    description=(
        "Examine the {service_title} graphQL service"
        "Use discovery techniques to discover the available queries for the GraphQL service"
    ),
    expected_output=(
        "An explanation of the main functions the GraphQL service provides"
        "and some examples that are inline with what the service offers {url}."
    ),
    agent=myAgent1,
    #context = [task2, task3],
    tools = [
        requestGraphqlQuery,
        spaceXSchema,
        spaceXExampleQuery
        ],
    human_input = True
)



# works with the user to understand there needs and query the service
task3 = Task(
    description=(
        "Understand the users request for {service_title}"
        "Identify possible GraphQL queries that may satisify the users needs"
    ),
    expected_output=(
        "Take the response form the GraphQL service and present it natural language"
        #"include the raw response formatted {url}."
    ),
    agent=myAgent3,
    context = [task1],
    tools = [
        requestGraphqlQuery,
        spaceXSchema,
        spaceXExampleQuery
        ],
    human_input = True
)

# trys different queries
task2 = Task(
    description=(
        "Examine the {service_title} graphQL service"
        "Use discovery techniques to discover the available queries that will answer the users query"
        "{user_input}"
    ),
    expected_output=(
        "An answer to the users query"
        #"list of graphql queries and their related input and outputs"
        #"plus a brief descrption for each of the queries {url}."
    ),
    agent=myAgent2,
    context = [
        task1,
        task3
        ],
    tools = [
        spaceXSchema,
        spaceXExampleQuery
        ],
    human_input = True
)

graphql_crew = Crew(
    agents=[
        myAgent1, 
        myAgent2, 
        myAgent3
        ],
    
    tasks=[
        task1,
        task2,
        task3
        ],
    
    manager_llm=llm,
    process=Process.hierarchical,
    embedder={
        "provider": "huggingface",
        "config": {
            #"model": "nomic-ai/nomic-embed-text-v1",
            "model": "mixedbread-ai/mxbai-embed-large-v1",  # Example model from HuggingFace
          }
         },
    cache=True,
    memory=True,
    output_log_file="graphqlAgentlog.txt",
    verbose = 2
)

print("Please enter your SpaceX query!!")
user_input = input()

inputs_obj = {
    "url": " http://docs.catalysis-hub.org/en/latest/tutorials/index.html#graphql",
    "service_description": "This is the SpaceX GraphQL service that contains interesting information about past flights",
    "service_title": "SpaceX",
    "user_input": user_input
}



result = graphql_crew.kickoff(inputs=inputs_obj)