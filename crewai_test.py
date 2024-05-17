import os
import platform
import subprocess
import requests
import re
import webbrowser
import shutil
import json
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process
from crewai_tools import BaseTool
from pydantic import BaseModel, Field

# Set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "NA"  # Replace "NA" with your actual OpenAI API key

class YouTubeSearchTool(BaseTool):
    name: str = "YouTube Search Tool"
    description: str = "Search for YouTube videos."

    def _run(self, search_query: str) -> str:
        youtube_search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
        response = requests.get(youtube_search_url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        scripts = soup.find_all('script')
        json_data = None
        for script in scripts:
            if script.string and 'ytInitialData' in script.string:
                json_text = script.string.strip()
                json_data = json.loads(json_text[json_text.find('{'): json_text.rfind('}') + 1])
                break

        if not json_data:
            raise ValueError("Failed to find ytInitialData in the page scripts.")

        video_url = None
        contents = json_data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents']
        for content in contents:
            items = content['itemSectionRenderer']['contents']
            for item in items:
                if 'videoRenderer' in item:
                    video_id = item['videoRenderer']['videoId']
                    video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
                    return video_url

        raise ValueError("No valid video URL found in the search results.")

def get_video_url_from_llm():
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            headers={"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"},
            json={
                "model": "llama3",
                "prompt": "What is the most popular song by Rick Astley on YouTube?",
                "stream": False,
                "temperature": 0.3
            }
        )
        response.raise_for_status()
        result = response.json()
        return extract_video_url(result.get('response', ''))
    except (requests.RequestException, ValueError) as e:
        print(f"An error occurred: {e}")
        return None

def extract_video_url(text):
    match = re.search(r'(https?://[^\s]+)', text)
    return match.group(0) if match else None

def open_url(video_url):
    if not video_url:
        print("No valid video URL to open.")
        return

    try:
        if platform.system() == 'Windows':
            subprocess.run(['cmd.exe', '/c', 'start', video_url], check=True)
        elif 'WSL_INTEROP' in os.environ:
            subprocess.run(['powershell.exe', '-Command', f'Start-Process "{video_url}"'], check=True)
        else:
            browser_path = next((shutil.which(browser) for browser in ['google-chrome', 'firefox', 'brave-browser'] if shutil.which(browser)), None)
            if browser_path:
                subprocess.run([browser_path, video_url], check=True)
            else:
                subprocess.run(['xdg-open', video_url], check=True)
    except subprocess.CalledProcessError:
        webbrowser.open(video_url, new=2)

class SearchAgentConfig(BaseModel):
    youtube_search_tool: YouTubeSearchTool

    class Config:
        arbitrary_types_allowed = True

class SearchAgent(Agent):
    youtube_search_tool: YouTubeSearchTool

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, youtube_search_tool: YouTubeSearchTool):
        super().__init__(role='Search Agent', goal='Search for the most popular song by Rick Astley on YouTube',
                         backstory='This agent is designed to find the most popular Rick Astley song on YouTube.',
                         tools=[youtube_search_tool])
        self.youtube_search_tool = youtube_search_tool

    def perform_task(self):
        video_url = get_video_url_from_llm()
        if not video_url:
            video_url = self.youtube_search_tool.run("Rick Astley Never Gonna Give You Up")
        return video_url

class BrowserAgent(Agent):
    def __init__(self):
        super().__init__(role='Browser Agent', goal='Open the browser and play the song',
                         backstory='This agent is responsible for opening the browser and playing the song.')

    def perform_task(self, video_url):
        open_url(video_url)

class PlayRickAstleySong(Task):
    def __init__(self):
        super().__init__(name='Play Rick Astley Song', description='Search for and play the most popular Rick Astley song on YouTube',
                         expected_output='The most popular Rick Astley song is played in the browser.')

    def execute(self, agents):
        search_agent, browser_agent = agents
        video_url = search_agent.perform_task()
        browser_agent.perform_task(video_url)

def main():
    youtube_search_tool = YouTubeSearchTool()
    search_agent = SearchAgent(youtube_search_tool=youtube_search_tool)
    browser_agent = BrowserAgent()
    play_rick_astley_song_task = PlayRickAstleySong()

    crew = Crew(agents=[search_agent, browser_agent], tasks=[play_rick_astley_song_task], verbose=2, process=Process.sequential)
    for task in crew.tasks:
        task.execute(crew.agents)

if __name__ == "__main__":
    main()
