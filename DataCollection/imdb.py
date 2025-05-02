import requests
import pandas as pd

OMDB_API_KEY = 'a86534ab'  # <-- Replace with your key
OMDB_BASE_URL = 'http://www.omdbapi.com/'

def get_movie_info_from_omdb(title):
    params = {
        'apikey': OMDB_API_KEY,
        't': title,
        'type': 'movie',
        'plot': 'full'  # Get full description
    }
    try:
        response = requests.get(OMDB_BASE_URL, params=params)
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
    except Exception:
        return None

def get_movies_data(titles):
    results = []
    failures = []
    for title in titles:
        info = get_movie_info_from_omdb(title)
        if info:
            results.append(info)
        else:
            failures.append(title)
    return results, failures

def search_not_present_movies(PresentDataFiles, Movies, FileName):
    # Combine all titles from PresentDataFiles into a set for quick lookup
    present_titles = []
    for datafile in PresentDataFiles:
        present_titles += list(pd.read_csv(datafile)["Title"])
    set_titles = set(present_titles)

    # Find movies that need to be searched for
    moviesToSearchFor = []
    alreadyPresent = [] 
    for movie in Movies :
        if movie not in set_titles:
            moviesToSearchFor.append(movie)
        else:
            alreadyPresent.append(movie)

    # Fetch data for movies that need to be searched
    successes, failures = get_movies_data(moviesToSearchFor)

    print("Failures:", failures)
    
    # Convert the successes to DataFrame
    result = pd.DataFrame(successes)
    
    # Remove rows with NaN values
    result.dropna(inplace=True)
    
    # Save the results to a CSV file
    result.to_csv(FileName, index=False)
    
    return {
        "Already Present":alreadyPresent,
        "Failures": failures
        }
    
def main():
    movies = ["Inception", "The Matrix", "Blade Runner"]
    save_movies_to_csv(movies, "imdb_movies.csv")

if __name__ == "__main__":
    main()
