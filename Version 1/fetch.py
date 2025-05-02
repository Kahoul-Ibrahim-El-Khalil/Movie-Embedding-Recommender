import pandas as pd
from imdb import IMDb

# Create IMDb instance
ia = IMDb()

# Get Top 250 movies
top_250 = ia.get_top250_movies()

# Prepare a list to store film data
film_data = []

for movie in top_250:
    movie_id = movie.movieID
    full_movie = ia.get_movie(movie_id)

    title = full_movie.get('title')
    year = full_movie.get('year')
    genres = ", ".join(full_movie.get('genres', []))
    plot = full_movie.get('plot outline') or full_movie.get('plot', [None])[0]

    film_data.append({
        'Title': title,
        'Year': year,
        'Genre': genres,
        'Description': plot
    })

# Create DataFrame and save as CSV
df = pd.DataFrame(film_data)
df.to_csv('top_250_imdb.csv', index=False)

print("CSV file saved as 'top_250_imdb.csv'")
