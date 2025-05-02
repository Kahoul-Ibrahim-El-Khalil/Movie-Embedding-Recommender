import pandas as pd

def segment_the_data_frame_by_years(Segment_Prefix, Data_Frame):
    # Sort the DataFrame
    Data_Frame.sort_values(by=["Year", "Title"], ascending=[False, True], inplace=True)
    
    # Drop rows where Year is NaN
    Data_Frame = Data_Frame[Data_Frame["Year"].notna()]

    # Convert year to integer
    Data_Frame["Year"] = Data_Frame["Year"].astype(int)
    
    # Get unique years
    years = Data_Frame["Year"].unique()
    
    paths = []
    for year in years:
        path = f"{Segment_Prefix}{year}.csv"
        Data_Frame[Data_Frame["Year"] == year].to_csv(path, index=False)
        paths.append(path)
    
    return paths
