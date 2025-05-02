

def prefix_match(df, reference_title):
    """
    Filters rows where the title starts with the same prefix (first word) as the reference_title.
    
    Args:
        df (pd.DataFrame): DataFrame containing a 'title' column.
        reference_title (str): The reference title to extract the prefix from.

    Returns:
        pd.DataFrame: Filtered DataFrame.
    """
    # Get first word (prefix) of the reference title
    prefix = reference_title.strip().split()[0].lower()
    
    # Return rows where title starts with the same prefix
    return df[df['Title'].str.lower().str.startswith(prefix)]
