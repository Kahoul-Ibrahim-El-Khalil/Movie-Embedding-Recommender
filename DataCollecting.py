import pandas as pd
import requests
from typing import List, Union, Optional, Dict
from urllib.parse import quote


OMDB_API_KEY = '********'  
OMDB_BASE_URL = 'http://www.omdbapi.com/'
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
TIMEOUT = 10

class BaseMovieSurfer:
    """Base class containing shared functionality"""
    def __init__(self, titles_to_ignore: List[str] = None):
        self.results: List[Dict[str, str]] = []
        self.results_titles: List[str] = []
        self.failures: List[str] = []
        self.dataframe = pd.DataFrame()
        self.titles_to_ignore = titles_to_ignore.copy() if titles_to_ignore else []
        
    def get_movies_data(self, titles_list: Union[str, List[str]]) -> None:
        """Fetch movie data for a list of titles"""
        if isinstance(titles_list, str):
            titles_list = [titles_list]
            
        for title in titles_list:
            info = self._get_movie_data(title)
            if info:
                self.results.append(info)
                self.results_titles.append(info["Title"])
            else:
                self.failures.append(title)

    def search_not_present_movies(self, search_titles: Union[str, List[str]]) -> None:
        """Search for movies not in results or ignore list"""
        if isinstance(search_titles, str):
            search_titles = [search_titles]

        for title in search_titles:
            if title not in (self.results_titles + self.titles_to_ignore):
                info = self._get_movie_data(title)
                if info:
                    self.results.append(info)
                    self.results_titles.append(title)
                    print(f"Added '{title}' to the results.")
                else:
                    print(f"Failed to retrieve information for '{title}'.")
            else:
                print(f"'{title}' is already known. Skipping...")

    def create_dataframe(self) -> None:
        """Convert results to cleaned DataFrame"""
        if self.results:
            self.dataframe = pd.DataFrame(self.results)
            self.dataframe = self.dataframe.dropna()
        else:
            print("No results to create DataFrame.")

    def write_to_csv(self, filename: str) -> None:
        """Write results to CSV file"""
        self.create_dataframe()
        if not self.dataframe.empty:
            self.dataframe.to_csv(filename, index=False)
            print(f"Results written to '{filename}'.")
            self.results = []
            self.dataframe = pd.DataFrame()
        else:
            print("No data to write to CSV.")

    def _get_movie_data(self, title: str) -> Optional[Dict[str, str]]:
        """To be implemented by child classes"""
        raise NotImplementedError


class IMDBSurfer(BaseMovieSurfer):
    def __init__(self, api_key: str = OMDB_API_KEY, titles_to_ignore: List[str] = None):
        super().__init__(titles_to_ignore)
        self.key = api_key

    def _get_movie_data(self, title: str) -> Optional[Dict[str, str]]:
        """Fetch movie info from OMDB API"""
        params = {
            'apikey': self.key,
            't': title,
            'type': 'movie',
            'plot': 'full'
        }
        try:
            response = requests.get(OMDB_BASE_URL, params=params, timeout=TIMEOUT)
            response.raise_for_status()
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


class WikipediaMovieSurfer(BaseMovieSurfer):
    def __init__(self, titles_to_ignore: List[str] = None):
        super().__init__(titles_to_ignore)

    def _get_movie_data(self, title: str) -> Optional[Dict[str, str]]:
        """Fetch structured movie data from Wikipedia and Wikidata"""
        # Step 1: Get Wikipedia page
        wiki_params = {
            'action': 'query',
            'format': 'json',
            'titles': f"{title} (film)",
            'prop': 'extracts|pageprops',
            'explaintext': True,
            'exintro': True
        }
        
        try:
            # Wikipedia API call
            wiki_response = requests.get(WIKIPEDIA_API_URL, params=wiki_params, timeout=10)
            wiki_response.raise_for_status()
            wiki_data = wiki_response.json()
            pages = wiki_data.get('query', {}).get('pages', {})
            page = next(iter(pages.values()))
            
            # Skip if not a film page
            if 'pageprops' not in page or 'wikibase_item' not in page['pageprops']:
                return None
                
            # Step 2: Get Wikidata
            wikidata_id = page['pageprops']['wikibase_item']
            wd_params = {
                'action': 'wbgetentities',
                'format': 'json',
                'ids': wikidata_id,
                'props': 'claims'
            }
            
            wd_response = requests.get(WIKIDATA_API_URL, params=wd_params, timeout=10)
            wd_response.raise_for_status()
            wd_data = wd_response.json()
            
            # Extract key film data
            claims = wd_data.get('entities', {}).get(wikidata_id, {}).get('claims', {})
            
            return {
                "Title": page.get('title', '').replace(' (film)', ''),
                "Year": self._extract_year(claims),
                "Director": self._extract_director(claims),
                "Genre": self._extract_genres(claims),
                "Description": page.get('extract', '')[:500] + "..." if len(page.get('extract', '')) > 500 else page.get('extract', ''),
                "Wikipedia_URL": f"https://en.wikipedia.org/wiki/{quote(page.get('title', ''))}"
            }
            
        except Exception as e:
            print(f"Error retrieving data for '{title}': {e}")
            return None

    def _extract_year(self, claims: Dict) -> str:
        """Extract release year from Wikidata claims"""
        if 'P577' in claims:
            date = claims['P577'][0].get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('time', '')
            return date[:4] if date else 'N/A'
        return 'N/A'

    def _extract_director(self, claims: Dict) -> str:
        """Extract director from Wikidata claims"""
        if 'P57' in claims:
            directors = []
            for director in claims['P57']:
                director_id = director.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id', '')
                if director_id:
                    name = self._get_entity_label(director_id)
                    if name:
                        directors.append(name)
            return ', '.join(directors) if directors else 'N/A'
        return 'N/A'

    def _extract_genres(self, claims: Dict) -> str:
        """Extract genres from Wikidata claims"""
        if 'P136' in claims:
            genres = []
            for genre in claims['P136']:
                genre_id = genre.get('mainsnak', {}).get('datavalue', {}).get('value', {}).get('id', '')
                if genre_id:
                    name = self._get_entity_label(genre_id)
                    if name:
                        genres.append(name)
            return ', '.join(genres) if genres else 'N/A'
        return 'N/A'

    def _get_entity_label(self, entity_id: str) -> Optional[str]:
        """Get label for a Wikidata entity"""
        params = {
            'action': 'wbgetentities',
            'format': 'json',
            'ids': entity_id,
            'props': 'labels',
            'languages': 'en'
        }
        try:
            response = requests.get(WIKIDATA_API_URL, params=params, timeout=5)
            response.raise_for_status()
            return response.json().get('entities', {}).get(entity_id, {}).get('labels', {}).get('en', {}).get('value')
        except:
            return None


def load_titles_from_csv(filepath: str) -> List[str]:
    """Helper function to load titles from CSV"""
    try:
        df = pd.read_csv(filepath)
        return df['Title'].tolist()
    except Exception as e:
        print(f"Error loading titles from {filepath}: {e}")
        return []
