import pandas as pd
import requests

OMDB_API_KEY = 'a86534ab'
OMDB_BASE_URL = 'http://www.omdbapi.com/'
TIMEOUT = 10

class IMDBSurfer:
    def __init__(self, ApiKey = OMDB_API_KEY, DataPath=None, Datasetscsvs=[]):
        self.key = ApiKey
        self.results = []
        self.results_titles = []
        self.data_path = DataPath
        self.datasets = Datasetscsvs
        self.failures = []

        self.available_titles = []
        for dataset in self.datasets:
            path = f"{self.data_path}/{dataset}" if self.data_path else dataset
            try:
                df = pd.read_csv(path)
                self.available_titles.extend(df["Title"].tolist())
            except Exception as e:
                print(f"IMDB Surfer Failed Failed to load dataset '{dataset}': {e}")

    def get_movie_info_from_omdb(self, Title):
        params = {
            'apikey': self.key,
            't': Title,
            'type': 'movie',
            'plot': 'full'
        }
        try:
            response = requests.get(OMDB_BASE_URL, params=params, timeout=TIMEOUT)
            data = response.json()
            if data.get('Response') == 'True':
                return {
                    "Title": data.get('Title', 'NaN'),
                    "Year": data.get('Year', 'NaN'),
                    "Genre": data.get('Genre', 'NaN'),
                    "Description": data.get('Plot', 'NaN')
                }
            else:
                return None
        except requests.exceptions.Timeout:
            print(f"Request for {Title} timed out. Skipping...")
            return None
        except Exception as e:
            print(f"Error retrieving {Title}: {e}")
            return None    
    
    def get_movies_data(self, TitlesList):
        for title in TitlesList:
            info = self.get_movie_info_from_omdb(title)
            if info:
                self.results.append(info)
            else:
                self.failures.append(title)
    
    def search_not_present_movies(self, SearchTitles):
        if isinstance(SearchTitles, str):
            SearchTitles = [SearchTitles]

        for title in SearchTitles:
            if title not in (self.results_titles + self.available_titles):
                info = self.get_movie_info_from_omdb(title)
                if info:
                    self.results.append(info)
                    self.results_titles.append(title)
                    print(f"Added {title} to the results.")
                else:
                    print(f"Failed to retrieve information for {title}.")
            else:
                print(f"{title} is already in the results. Skipping...")

    
    def write_to_csv(self, FileName):
        dataFrame = pd.DataFrame(self.results)
        dataFrame = dataFrame.dropna()
        dataFrame.to_csv(FileName, index=False)
        print(f"Results written to {FileName}.")
        self.datasets.append(FileName)
        self.results = []
