import pandas as pd
import DataCollection.imdb



dataFiles = [
    "DataCollection/PostTokenzing/45000-cleaned.csv",
    "DataCollection/PostTokenzing/custom-imdb.csv",
    "DataCollection/PostTokenzing/X-Men.csv"
]

imdb.search_not_present_movies()