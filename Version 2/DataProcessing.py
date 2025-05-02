import pandas as pd
from TextProcessing import TextProcessor  # Make sure file is named `text_processing.py`

class DataProcessor:
    def __init__(self, text_processor, data_path=None, data_csv=""):
        self.text_processor = text_processor
        self.data_path = data_path
        self.data_csv = data_csv
        self.dataframe = pd.DataFrame()

    def load_dataframe(self):
        try:
            filepath = self.data_csv if self.data_path is None else f"{self.data_path}/{self.data_csv}"
            self.dataframe = pd.read_csv(filepath)
            print(f"Data Processor DataFrame loaded from: {filepath}")
        except Exception as ex:
            print(f"Data Processor Failed to load DataFrame from {filepath}: {ex}")

    def search_by_prefix_of_title(self, reference_title):
        if self.dataframe.empty:
            print("DataFrame is empty. Load it first.")
            return pd.DataFrame()
        prefix = reference_title.strip().split()[0].lower()
        return self.dataframe[self.dataframe['Title'].str.lower().str.startswith(prefix)]

    def add_rows_of_dataframe(self, new_df):
        if not isinstance(new_df, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")
        if self.dataframe.empty:
            self.dataframe = new_df.copy()
        else:
            self.dataframe = pd.concat([self.dataframe, new_df], ignore_index=True)
        print("Rows added successfully.")

    def determine_keywords(self):
        """Fill NaN or empty 'Keywords' with processed results from 'Description'."""
        print(self.dataframe.columns)
        if 'Description' not in self.dataframe.columns:
            print("Missing required columns: 'Description'.")
            return
        # Ensure 'Keywords' column exists. If not, initialize it with NaN
        if 'Keywords' not in self.dataframe.columns:
            self.dataframe['Keywords'] = None  # Use None to indicate empty lists initially
    
        for index, row in self.dataframe.iterrows():
            # Check if Keywords is NaN, None, or empty
            if pd.isna(row['Keywords']) or row['Keywords'] == None or row['Keywords'] == []:
                description = str(row['Description'])
                keywords = self.text_processor.extract_keywords(description)
                self.dataframe.at[index, 'Keywords'] = keywords  # Assign the list of keywords
    
        print("Keyword extraction completed.")

    def sync_dataframe_with_dataset(self):
        try:
            filepath = self.data_csv if self.data_path is None else f"{self.data_path}/{self.data_csv}"
            self.dataframe.to_csv(filepath, index=False)
            print(f"DataFrame synced to {filepath}")
        except Exception as ex:
            print(f"Failed to write DataFrame to CSV: {ex}")
