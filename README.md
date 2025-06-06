# 🎬 A Data Science Project: Movie Predictor

This repository contains a data science project built around predicting and analyzing movie-related data sourced from IMDB.

## 📁 Project Structure

- **`data/`**  
  Contains raw `.csv` data files collected from IMDB.

- **`DataCollecting.py`**  
  This module is responsible for fetching and storing movie data from the IMDB API or web scraping.

- **`MoviePredictor.py`**  
  The core logic of the project, including preprocessing, model training, evaluation, and prediction.

## 📓 Notebooks

- **`IMDB-surfer.ipynb`**  
  An interactive notebook for exploring and visualizing the movie dataset.

- **`main.ipynb`**  
  The main entry point to run the end-to-end pipeline.

> The project is designed to be executed in a **Jupyter Notebook environment**, allowing for interactive data analysis and model experimentation.

## 🧠 Embeddings

For convenience, embeddings are saved to disk. You can either reuse the stored versions or let the code recompute them when necessary

## 📦 Dependencies

- `numpy`
- `pandas`
- `scikit-learn` (`sklearn`)
- `plotly`
- `joblib`
- `requests`
