import TextProcessing
import DataCollecting
import DataProcessing

import warnings
warnings.filterwarnings("ignore")

DATASET = "data.csv"

tp = TextProcessing.TextProcessor()

imdb = DataCollecting.IMDBSurfer(DataCollecting.OMDB_API_KEY, None, [DATASET])

dp = DataProcessing.DataProcessor(tp, None, DATASET)

print(tp.extract_keywords("Hello Monica, this is a good day, I am trying my best here"))