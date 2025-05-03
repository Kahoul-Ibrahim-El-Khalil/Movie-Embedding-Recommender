import pandas as pd
import requests
from typing import List, Union, Optional, Dict

OMDB_API_KEY = 'a86534ab'  # Note: Consider using environment variables for API keys
OMDB_BASE_URL = 'http://www.omdbapi.com/'
TIMEOUT = 10

class IMDBSurfer:
    def __init__(self, api_key: str = OMDB_API_KEY, data_path: Optional[str] = None, datasets_csvs: List[str] = []):
        """
        Initialize the IMDBSurfer with API key, data path, and dataset CSV files.
        
        Args:
            api_key (str): OMDB API key. Defaults to OMDB_API_KEY.
            data_path (Optional[str]): Path to directory containing datasets. Defaults to None.
            datasets_csvs (List[str]): List of dataset CSV filenames. Defaults to empty list.
        """
        self.key = api_key
        self.results: List[Dict[str, str]] = []
        self.results_titles: List[str] = []
        self.data_path = data_path
        self.datasets = datasets_csvs.copy()  # Avoid mutable default argument
        self.failures: List[str] = []
        self.dataframe = pd.DataFrame()
        self.available_titles: List[str] = []
        
        # Load datasets if provided
        for dataset in self.datasets:
            path = f"{self.data_path}/{dataset}" if self.data_path else dataset
            try:
                df = pd.read_csv(path)
                self.available_titles.extend(df["Title"].tolist())
            except Exception as e:
                print(f"IMDBSurfer: Failed to load dataset '{dataset}': {e}")

    def get_movie_info_from_omdb(self, title: str) -> Optional[Dict[str, str]]:
        """
        Fetch movie info from OMDB API.
        
        Args:
            title (str): Movie title to search for.
            
        Returns:
            Optional[Dict[str, str]]: Movie info if found, None otherwise.
        """
        params = {
            'apikey': self.key,
            't': title,
            'type': 'movie',
            'plot': 'full'
        }
        try:
            response = requests.get(OMDB_BASE_URL, params=params, timeout=TIMEOUT)
            response.raise_for_status()  # Raise HTTPError for bad responses
            data = response.json()
            
            if data.get('Response') == 'True':
                return {
                    "Title": data.get('Title', 'N/A'),
                    "Year": data.get('Year', 'N/A'),
                    "Genre": data.get('Genre', 'N/A'),
                    "Description": data.get('Plot', 'N/A')
                }
            return None
        except requests.exceptions.Timeout:
            print(f"Request for '{title}' timed out. Skipping...")
            return None
        except requests.exceptions.RequestException as e:
            print(f"Error retrieving '{title}': {e}")
            return None    

    def get_movies_data(self, titles_list: Union[str, List[str]]) -> None:
        """
        Fetch movie data for a list of titles and store results.
        
        Args:
            titles_list (Union[str, List[str]]): Single title or list of titles.
        """
        if isinstance(titles_list, str):
            titles_list = [titles_list]
            
        for title in titles_list:
            info = self.get_movie_info_from_omdb(title)
            if info:
                self.results.append(info)
                self.results_titles.append(info["Title"])  # Track successful titles
            else:
                self.failures.append(title)

    def search_not_present_movies(self, search_titles: Union[str, List[str]]) -> None:
        """
        Search for movies not already in results or available datasets.
        
        Args:
            search_titles (Union[str, List[str]]): Single title or list of titles.
        """
        if isinstance(search_titles, str):
            search_titles = [search_titles]

        for title in search_titles:
            if title not in (self.results_titles + self.available_titles):
                info = self.get_movie_info_from_omdb(title)
                if info:
                    self.results.append(info)
                    self.results_titles.append(title)
                    print(f"Added '{title}' to the results.")
                else:
                    print(f"Failed to retrieve information for '{title}'.")
            else:
                print(f"'{title}' is already in the results. Skipping...")

    def create_dataframe(self) -> None:
        """Convert collected results to a cleaned DataFrame."""
        if self.results:
            self.dataframe = pd.DataFrame(self.results)
            self.dataframe = self.dataframe.dropna()
        else:
            print("No results to create DataFrame.")

    def write_to_csv(self, filename: str) -> None:
        """
        Write results to a CSV file.
        
        Args:
            filename (str): Output CSV filename.
        """
        self.create_dataframe()
        if not self.dataframe.empty:
            self.dataframe.to_csv(filename, index=False)
            print(f"Results written to '{filename}'.")
            self.datasets.append(filename)
            self.results = []  # Clear results after writing
            self.dataframe = pd.DataFrame()  # Reset DataFrame
        else:
            print("No data to write to CSV.")


# Example usage
if __name__ == "__main__":
    surfer = IMDBSurfer(datasets_csvs=["movies.csv"])
    
    # Fetch movie data
    surfer.get_movies_data(["Inception", "The Matrix"])
    
    # Search for new movies
    surfer.search_not_present_movies(["Interstellar", "Inception"])  # Inception will be skipped
    
    # Save to CSV
    surfer.write_to_csv("output.csv")