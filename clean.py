import pandas as pd

# Removed unwanted columns, removed duplicate rows, removed rows with missing values, removed rows with at least two `0` values. 
# The cleaned dataset is saved to a new CSV file.

# Load the dataset (update the file path as needed)
data = pd.read_csv('more_cleaned_artists.csv')

# Remove unwanted columns
columns_to_remove = ['tags_mb', 'artist_mb', 'tags_mb', 'ambiguous_artist', 'country_mb' ]
data = data.drop(columns=columns_to_remove, errors='ignore')

# Remove duplicate rows based on the `artist_lastfm` column
data = data.drop_duplicates(subset=['artist_lastfm'])

# Remove rows with missing values
data = data.dropna()


def has_fewer_than_two_zeros(row):
    return (row == 0).sum() < 2

# Remove rows with at least two `0` values
data = data[data.apply(has_fewer_than_two_zeros, axis=1)]

# Function to check if text contains only ASCII characters
def is_ascii(text):
    return all(ord(c) < 128 for c in str(text))

# List of text columns to check
text_columns = ['artist_lastfm']  # Add other text columns if needed

# Remove rows with non-ASCII characters in any of the specified columns
for col in text_columns:
    data = data[data[col].apply(is_ascii)]

# Save the cleaned dataset
cleaned_file_path = 'more_cleaned_dataset.csv'
data.to_csv(cleaned_file_path, index=False)

print(f"Cleaned dataset saved to {cleaned_file_path}. Remaining rows: {len(data)}")
