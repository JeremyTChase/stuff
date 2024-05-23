import os
import requests
from textwrap import dedent
from crewai import Crew, Agent, Task, Process 
from langchain_community.llms import Ollama
from crewai_tools import tool

llm = Ollama(
    model="mistral"
    )

@tool
def GetGraphQLSchema()->str:
    """Useful for understanding what services and features a GraphQL service has to offer"""
    query = """   
            {
                __schema {
                    types {
                        name
                        fields {
                            name
                        }
                    }
                }
            }"""
            

        # Define the GraphQL endpoint
    url = "https://main--spacex-l4uc6p.apollographos.net/graphql"

    # Define the introspection query
    introspection_query = """
    {
    __schema {
        queryType {
        name
        }
        mutationType {
        name
        }
        subscriptionType {
        name
        }
        types {
        ...FullType
        }
        directives {
        name
        description
        locations
        args {
            ...InputValue
        }
        }
    }
    }

    fragment FullType on __Type {
    kind
    name
    description
    fields(includeDeprecated: true) {
        name
        description
        args {
        ...InputValue
        }
        type {
        ...TypeRef
        }
        isDeprecated
        deprecationReason
    }
    inputFields {
        ...InputValue
    }
    interfaces {
        ...TypeRef
    }
    enumValues(includeDeprecated: true) {
        name
        description
        isDeprecated
        deprecationReason
    }
    possibleTypes {
        ...TypeRef
    }
    }

    fragment InputValue on __InputValue {
    name
    description
    type {
        ...TypeRef
    }
    defaultValue
    }

    fragment TypeRef on __Type {
    kind
    name
    ofType {
        kind
        name
        ofType {
        kind
        name
        ofType {
            kind
            name
        }
        }
    }
    }
    """

    # Define the headers
    headers = {
        "Content-Type": "application/json"
    }

    # Define the payload
    payload = {
        "query": introspection_query
    }

    # Make the request
    response = requests.post(url, json=payload, headers=headers)

    # Print the response
    print(response.json())
    return response.json()

myAgent1 = Agent(
			role="Lead GraphQL Engineer",
			goal=dedent("""\
				develop plans based on user input that
                need graphql query support
               """),
			backstory=dedent("""\
				Lead GraphQL engineer with 20 years experience in developing
                    GraphQL services"""),
			tools=[
					GetGraphQLSchema
			],
			allow_delegation=False,
			llm=llm,
			verbose=True
		)

# Task for Data Analyst Agent:
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
    agent=myAgent1,
)

graphql_crew = Crew(
    agents=[myAgent1],
    
    tasks=[GraphQLServiceAnalysis],
    
    manager_llm=llm,
    process=Process.sequential,
    # embedder={
    #     "provider": "huggingface",
    #     "config": {
    #         "model": "mixedbread-ai/mxbai-embed-large-v1",  # Example model from HuggingFace
    #       }
    #     },
    memory=False,
    output_log_file="graphqlAgentlog.txt",
    verbose = 2
)

inputs_obj = {
    "url": " http://docs.catalysis-hub.org/en/latest/tutorials/index.html#graphql"
}


result = graphql_crew.kickoff(inputs=inputs_obj)