import os
import platform
import subprocess
import requests
import re
import json
import shutil
import webbrowser
from bs4 import BeautifulSoup
from crewai import Agent, Task, Crew, Process

# Set the OPENAI_API_KEY environment variable
os.environ["OPENAI_API_KEY"] = "NA"  # Replace "NA" with your actual OpenAI API key

# Define the search agent
class SearchAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Search Agent',
            goal='Search for the most popular song by Rick Astley on YouTube',
            backstory='This agent is designed to find the most popular Rick Astley song on YouTube.'
        )

    def perform_task(self):
        video_url = self.get_video_url_from_llm()
        if not video_url:
            video_url = self.get_video_url_from_web()
        return video_url

    def get_video_url_from_llm(self):
        try:
            # Use the local LLM to search for Rick Astley's most popular song
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
            response.raise_for_status()  # Ensure the request was successful
            try:
                result = response.json()
                result_text = result.get('response', '')
                video_url = self.extract_video_url(result_text)
                if video_url:
                    print(f"Found URL via LLM: {video_url}")
                return video_url
            except ValueError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Raw response text: {response.text}")
                return None
        except (requests.RequestException, ValueError) as e:
            print(f"An error occurred: {e}")
            return None

    def get_video_url_from_web(self):
        try:
            # Perform a YouTube search for Rick Astley's most popular song
            search_query = "Rick Astley Never Gonna Give You Up"
            youtube_search_url = f"https://www.youtube.com/results?search_query={search_query.replace(' ', '+')}"
            response = requests.get(youtube_search_url)
            response.raise_for_status()

            # Parse the search results page
            soup = BeautifulSoup(response.text, 'html.parser')
            scripts = soup.find_all('script')
            json_data = None
            for script in scripts:
                if script.string and 'ytInitialData' in script.string:
                    json_text = script.string.strip()
                    json_data = json.loads(json_text[json_text.find('{'): json_text.rfind('}') + 1])
                    break

            if json_data is None:
                raise ValueError("Failed to find ytInitialData in the page scripts.")

            video_url = None
            contents = json_data['contents']['twoColumnSearchResultsRenderer']['primaryContents']['sectionListRenderer']['contents']
            for content in contents:
                items = content['itemSectionRenderer']['contents']
                for item in items:
                    if 'videoRenderer' in item:
                        video_id = item['videoRenderer']['videoId']
                        video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
                        print(f"Found URL via Web: {video_url}")
                        return video_url

            raise ValueError("No valid video URL found in the search results.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None

    def extract_video_url(self, text):
        # Extract the YouTube URL from the LLM response
        print(f"Extracting URL from text: {text}")
        match = re.search(r'(https?://[^\s]+)', text)
        if match:
            print(f"Found URL: {match.group(0)}")
        else:
            print("No URL found.")
        return match.group(0) if match else None

# Define the browser agent
class BrowserAgent(Agent):
    def __init__(self):
        super().__init__(
            role='Browser Agent',
            goal='Open the browser and play the song',
            backstory='This agent is responsible for opening the browser and playing the song.'
        )

    def perform_task(self, video_url):
        if video_url:
            try:
                if platform.system() == 'Windows':
                    print("Attempting to open URL with cmd.exe")
                    subprocess.run(['cmd.exe', '/c', 'start', video_url], check=True)
                    print(f"Opened URL with cmd.exe: {video_url}")
                elif 'WSL_INTEROP' in os.environ:
                    print("Attempting to open URL with Windows browser from WSL using PowerShell")
                    self.open_with_powershell(video_url)
                else:
                    # Check for common browsers and use the first one found
                    browsers = ['google-chrome', 'firefox', 'brave-browser']
                    browser_path = next((shutil.which(browser) for browser in browsers if shutil.which(browser)), None)
                    
                    if browser_path:
                        print(f"Attempting to open URL with {browser_path}")
                        subprocess.run([browser_path, video_url], check=True)
                        print(f"Opened URL with {browser_path}: {video_url}")
                    else:
                        print("Attempting to open URL with xdg-open")
                        try:
                            subprocess.run(['xdg-open', video_url], check=True)
                            print(f"Opened URL with xdg-open: {video_url}")
                        except subprocess.CalledProcessError as e:
                            print(f"xdg-open failed: {e}")
                            print("Attempting to open URL with webbrowser.open")
                            webbrowser.open(video_url, new=2)
                            print(f"Opened URL with webbrowser.open: {video_url}")
            except subprocess.CalledProcessError as e:
                print(f"Failed to open URL: {e}")
                print("Attempting to open URL with webbrowser.open as last resort")
                webbrowser.open(video_url, new=2)
                print(f"Opened URL with webbrowser.open: {video_url}")
        else:
            print("No valid video URL to open.")

    def open_with_powershell(self, url):
        try:
            powershell_command = f'Start-Process "{url}"'
            subprocess.run(['powershell.exe', '-Command', powershell_command], check=True)
            print(f"Opened URL with PowerShell: {url}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to open URL with PowerShell: {e}")
            print("Attempting to open URL with webbrowser.open as last resort")
            webbrowser.open(url, new=2)
            print(f"Opened URL with webbrowser.open: {url}")

# Define the task
class PlayRickAstleySong(Task):
    def __init__(self):
        super().__init__(
            name='Play Rick Astley Song',
            description='Search for and play the most popular Rick Astley song on YouTube',
            expected_output='The most popular Rick Astley song is played in the browser.'
        )

    def execute(self, agents):
        search_agent = agents[0]
        browser_agent = agents[1]

        # Search for the song
        video_url = search_agent.perform_task()

        # Play the song
        browser_agent.perform_task(video_url)

def main():
    # Create the agents
    search_agent = SearchAgent()
    browser_agent = BrowserAgent()

    # Create the task
    play_rick_astley_song_task = PlayRickAstleySong()

    # Create the Crew with agents and tasks
    crew = Crew(
        agents=[search_agent, browser_agent],
        tasks=[play_rick_astley_song_task],
        verbose=2,
        process=Process.sequential
    )

    # Execute the task manually with agents
    for task in crew.tasks:
        task.execute(crew.agents)

if __name__ == "__main__":
    main()