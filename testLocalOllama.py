from crewai import Agent, Task, Crew
from langchain_openai import ChatOpenAI
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = "NA"
#os.environ["OPENAI_MODEL_NAME"]="gpt-3.5-turbo-0125"
#os.environ["OPENAI_API_BASE"]="https://api.openai.com/v1"

# Configure the language model
llm = ChatOpenAI(
    model="llama3:8b",
    base_url="http://localhost:11434/v1"
)

# Define the agent
general_agent = Agent(
    role="Math Professor",
    goal="Provide the solution to the students that are asking mathematical questions and give them the answer.",
    backstory="You are an excellent math professor that likes to solve math questions in a way that everyone can understand your solution",
    allow_delegation=False,
    verbose=True,
    llm=llm
)

# Define a simple task
task = Task(
    description="What is 3 + 5?",
    agent=general_agent,
    expected_output="A numerical answer."
)

# Create the crew
crew = Crew(
    agents=[general_agent],
    tasks=[task],
    verbose=2,
    max_rpm=100,
    output_log_file="crew_output.log"  # Ensure this is set correctly
)

# Kick off the crew and print the result
try:
    result = crew.kickoff()
    print("######################")
    print(result)
except Exception as e:
    print(f"An error occurred: {e}")

# Access and print the task output
print(f"Task Output: {task.output}")