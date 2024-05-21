import json
import os
from crewai import Agent, Task, Crew, Process
#from crewai_tools import SerperDevTool, DirectoryReadTool, FileReadTool
from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = "NA"  # Replace with your actual OpenAI API key
os.environ["OPENAI_MODEL_NAME"] = "llama3:8b"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"

# Initialize the LLM
llm = ChatOpenAI(
    model="llama3:8b",
    base_url="http://localhost:11434/v1"
)

# Directory containing JSON files
json_dir = 'agents'

# Load JSON files
def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Load agents
agents = {}
for file_name in os.listdir(json_dir):
    if file_name.startswith('agent') and file_name.endswith('.json'):
        agent_config = load_json(os.path.join(json_dir, file_name))
        tools = [globals()[tool]() for tool in agent_config['tools']]
        agents[agent_config['role']] = Agent(
            role=agent_config['role'],
            goal=agent_config['goal'],
            backstory=agent_config['backstory'],
            tools=tools,
            verbose=agent_config['verbose']
        )

# Load tasks
tasks = []
for file_name in os.listdir(json_dir):
    if file_name.startswith('task') and file_name.endswith('.json'):
        task_config = load_json(os.path.join(json_dir, file_name))
        task_agent = agents[task_config['agent']]
        tools = [globals()[tool]() for tool in task_config['tools']]
        task = Task(
            description=task_config['description'],
            expected_output=task_config['expected_output'],
            agent=task_agent,
            tools=tools,
            async_execution=task_config.get('async_execution', False),
            output_file=task_config.get('output_file', None)
        )
        tasks.append(task)

# Load crew configuration
crew_config = load_json(os.path.join(json_dir, 'crew.json'))

# Create crew
crew = Crew(
    agents=[agents[role] for role in crew_config['agents']],
    tasks=tasks,
    process=Process[crew_config['process']],
    verbose=crew_config['verbose'],
    memory=True,
    cache=True,
    embedder={
        "provider": "huggingface",
        "config": {
            "model": "mixedbread-ai/mxbai-embed-large-v1",  # Example model from HuggingFace
        }}
)

# Execute tasks
result = crew.kickoff()
print(result)