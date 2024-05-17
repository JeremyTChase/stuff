import requests

url = "https://main--spacex-l4uc6p.apollographos.net/graphql"
query = """
{
  launchesPast(limit: 5) {
    mission_name
    launch_date_utc
  }
}
"""

response = requests.post(url, json={'query': query})
if response.status_code == 200:
    print(response.json())
else:
    print(f"Query failed to run by returning code of {response.status_code}. {response.text}")