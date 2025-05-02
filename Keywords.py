import pandas as pd
import DataProcessing.tokenizing as tokenizing

# Read and concatenate CSV files
aggregated = pd.concat([
    pd.read_csv("DataCollection/PreTokenizing/HighestGrossing.csv"),
    pd.read_csv("DataCollection/PreTokenizing/historical_film_list.csv"),
    pd.read_csv("DataCollection/PreTokenizing/PopCulture.csv"),
    pd.read_csv("DataCollection/PreTokenizing/TheLastOfTheMohicans.csv")
])

# Save the concatenated DataFrame to a CSV file
aggregated.to_csv("DataCollection/PreTokenizing/aggregated.csv", index=False)

# Tokenize the segments using the newly created file
tokenizing.tokenize_segments(["DataCollection/PreTokenizing/aggregated.csv"], inplace=False)
