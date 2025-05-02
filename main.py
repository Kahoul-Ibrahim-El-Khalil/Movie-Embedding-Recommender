import pandas as pd
import DataCollection.imdb

popular_movies = [
    "The Shawshank Redemption",
    "The Godfather",
    "The Dark Knight",
    "Pulp Fiction",
    "Forrest Gump",
    "Inception",
    "The Matrix",
    "Star Wars: A New Hope",
    "The Lord of the Rings: The Fellowship of the Ring",
    "Titanic",
    "Avatar",
    "Back to the Future",
    "Jurassic Park",
    "The Avengers",
    "The Lion King",
    "Fight Club",
    "Goodfellas",
    "The Godfather: Part II",
    "Gladiator",
    "The Empire Strikes Back",
    "The Departed",
    "The Silence of the Lambs",
    "Interstellar",
    "The Princess Bride",
    "The Exorcist",
    "Blade Runner",
    "The Terminator",
    "Jaws",
    "The Dark Knight Rises",
    "The Big Lebowski",
    "Schindler's List"
]

dataFiles = [
    "DataCollection/PostTokenzing/"
]
imdb.search_not_present_movies()