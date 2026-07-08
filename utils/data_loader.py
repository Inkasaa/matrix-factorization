import os
import pandas as pd

def load_ratings(data_dir="data"):
    """Loads the ratings.dat file into a Pandas DataFrame."""
    path = os.path.join(data_dir, "ratings.dat")
    columns = ["user_id", "movie_id", "rating", "timestamp"]
    # sep="::" requires engine="python" as the separator is longer than 1 character.
    df = pd.read_csv(path, sep="::", names=columns, engine="python", encoding="ISO-8859-1")
    return df

def load_movies(data_dir="data"):
    """Loads the movies.dat file into a Pandas DataFrame."""
    path = os.path.join(data_dir, "movies.dat")
    columns = ["movie_id", "title", "genres"]
    df = pd.read_csv(path, sep="::", names=columns, engine="python", encoding="ISO-8859-1")
    return df

def load_users(data_dir="data"):
    """Loads the users.dat file into a Pandas DataFrame."""
    path = os.path.join(data_dir, "users.dat")
    columns = ["user_id", "gender", "age", "occupation", "zip_code"]
    df = pd.read_csv(path, sep="::", names=columns, engine="python", encoding="ISO-8859-1")
    return df

def load_all_data(data_dir="data"):
    """Loads all three files and prints data shapes for validation."""
    print("🔄 Loading MovieLens 1M dataset...")
    
    ratings = load_ratings(data_dir)
    movies = load_movies(data_dir)
    users = load_users(data_dir)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Ratings: {ratings.shape[0]:,}")
    print(f"   - Movies:  {movies.shape[0]:,}")
    print(f"   - Users:   {users.shape[0]:,}")
    
    return ratings, movies, users

if __name__ == "__main__":
    # Test execution to verify paths and file integrity
    try:
        ratings, movies, users = load_all_data()
        print("\nPreview of the ratings dataset:")
        print(ratings.head())
    except FileNotFoundError as e:
        print(f"❌ Error: Dataset file not found. Make sure the .dat files are placed inside the '{data_dir}/' directory.")