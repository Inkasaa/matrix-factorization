import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.data_loader import load_ratings

def create_and_process_matrices(data_dir="data", output_dir="processed"):
    """
    Creates user-item interaction matrices, splits into train/test,
    normalizes the matrix, and saves it.
    """
    # 1. Load data
    ratings_df = load_ratings(data_dir)
    
    # 2. Train-Test Split (fixed random_state=42)
    train_df, test_df = train_test_split(
        ratings_df, test_size=0.2, random_state=42
    )
    print(f"📊 Data split finished: Train size={train_df.shape[0]:,}, Test size={test_df.shape[0]:,}")

    # Determine unique user and movie profiles to ensure identical matrix shapes
    all_users = np.sort(ratings_df["user_id"].unique())
    all_movies = np.sort(ratings_df["movie_id"].unique())
    
    # Map raw IDs to sequential 0-indexed positions
    user_to_idx = {uid: i for i, uid in enumerate(all_users)}
    movie_to_idx = {mid: i for i, mid in enumerate(all_movies)}
    
    num_users = len(all_users)
    num_movies = len(all_movies)
    
    # 3. Create Interaction Matrices
    # Initialize unobserved interactions as 0
    R_train = np.zeros((num_users, num_movies))
    R_test = np.zeros((num_users, num_movies))
    
    # Populate the train matrix
    for row in train_df.itertuples():
        u_idx = user_to_idx[row.user_id]
        m_idx = movie_to_idx[row.movie_id]
        R_train[u_idx, m_idx] = row.rating
        
    # Populate the test matrix
    for row in test_df.itertuples():
        u_idx = user_to_idx[row.user_id]
        m_idx = movie_to_idx[row.movie_id]
        R_test[u_idx, m_idx] = row.rating
        
    # Create masks indicating where real entries exist
    train_mask = R_train > 0
    test_mask = R_test > 0
    
    # 4. Normalization (Mean-centering based on observed training user ratings)
    print("⚖️ Normalizing user-item interaction matrix...")
    R_train_norm = R_train.copy()
    user_means = np.zeros(num_users)
    
    for u in range(num_users):
        rated_indices = train_mask[u]
        if np.any(rated_indices):
            user_means[u] = np.mean(R_train[u, rated_indices])
            # Center only the observed ratings
            R_train_norm[u, rated_indices] -= user_means[u]
            
    # 5. Export processed elements to disk
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training matrix to specified csv format path
    df_export = pd.DataFrame(R_train_norm, index=all_users, columns=all_movies)
    matrix_path = os.path.join(output_dir, "user_item_matrix.csv")
    df_export.to_csv(matrix_path)
    print(f"💾 Normalized user-item matrix saved to: {matrix_path}")
    
    return {
        "R_train": R_train,
        "R_train_norm": R_train_norm,
        "R_test": R_test,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "user_means": user_means,
        "user_to_idx": user_to_idx,
        "movie_to_idx": movie_to_idx
    }

if __name__ == "__main__":
    try:
        matrices = create_and_process_matrices()
        print("✅ Matrix processing pipeline execution completed without errors!")
    except Exception as e:
        print(f"❌ Execution failed: {e}")