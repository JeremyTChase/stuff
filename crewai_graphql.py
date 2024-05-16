import crewai
import requests
from crewai import Task

class GraphQLAgent(crewai.Agent):
    def __init__(self, name, api_url, headers=None):
        super().__init__(name)
        self.api_url = api_url
        self.headers = headers if headers else {}

    def query(self, query, variables=None):
        response = requests.post(
            self.api_url,
            json={'query': query, 'variables': variables},
            headers=self.headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed with status code {response.status_code}")

class LaunchesAgent(GraphQLAgent):
    def __init__(self, api_url, headers=None):
        super().__init__("LaunchesAgent", api_url, headers)
    
    def get_launches(self, limit):
        query = """
        query GetLaunches($limit: Int) {
            launchesPast(limit: $limit) {
                mission_name
                launch_date_utc
                rocket {
                    rocket_name
                }
                launch_site {
                    site_name_long
                }
            }
        }
        """
        variables = {"limit": limit}
        return self.query(query, variables)
    

class SpaceXCrew(crewai.Crew):
    def __init__(self, api_url, headers=None):
        agents = []
        tasks = []
        config = {}  # Optional configuration

        super().__init__(agents=agents, tasks=tasks, config=config)
        
        self.launches_agent = LaunchesAgent(api_url, headers)
        self.add_agent(self.launches_agent)

        # Populate the agents list after adding them
        self.agents.append(self.launches_agent)

    def get_recent_launches(self, limit=5):
        task = LaunchesTask(self.launches_agent, limit)
        self.add_task(task)
        return task.run()



if __name__ == "__main__":
    api_url = "https://api.spacex.land/graphql/"
    headers = {"Content-Type": "application/json"}

    space_x_crew = SpaceXCrew(api_url, headers)

    # Get recent launches
    try:
        recent_launches = space_x_crew.get_recent_launches(limit=5)
        print("Recent Launches:")
        for launch in recent_launches['data']['launchesPast']:
            print(f"Mission: {launch['mission_name']}")
            print(f"Date: {launch['launch_date_utc']}")
            print(f"Rocket: {launch['rocket']['rocket_name']}")
            print(f"Launch Site: {launch['launch_site']['site_name_long']}")
            print("-----")
    except Exception as e:
        print(f"An error occurred: {e}")
