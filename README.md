# 🎬 Collaborative Filtering Movie Recommender (SVD vs PMF)

An advanced collaborative filtering movie recommendation engine designed to benchmark a baseline **Singular Value Decomposition (SVD)** model against a high-performance **Probabilistic Matrix Factorization (PMF)** model using the MovieLens 1M dataset. 

It includes an interactive, premium Streamlit dashboard with custom user profiling, multi-genre hybrid blending, and detailed local model interpretability.

---

## 🌟 Key Features

* **Advanced Matrix Factorization Engines**:
  * **Biased SVD**: Implements stochastic gradient descent (SGD) with adaptive learning rate decay.
  * **Vectorized PMF**: Formulates matrix factorization as a probabilistic graphical model, utilizing fast vectorized batch updates (processing up to 50,000 ratings at a time) for rapid training.
* **Interactive Streamlit Dashboard (`app.py`)**:
  * **Top & Worst Recommendations**: Inspect the top $N$ recommendations or flip the view to see the worst $N$ recommendations (up to 100 movies).
  * **Activity Profiles**: Visually review existing users' historical review volume, genre preferences, and full review history.
  * **Custom Taste Profiler Sandbox**: Allows visitors to create a brand new profile on-the-fly by rating a subset of movies (1-5★) and generates personalized recommendations instantly.
  * **Hybrid Blend Engine**: Includes an experimental mode that blends model predictions with content-aware genre boosts based on user taste history.
* **Explanations & Documentation**: Includes a `notes/` directory with detailed, non-technical explanations for both models.

---

## 📊 Model Performance Comparison

The models are evaluated using **RMSE (Root Mean Squared Error)** on a 20% validation test set:

| Model | Test RMSE | Description |
| :--- | :--- | :--- |
| **SVD Baseline** | `0.8842` | Classically tuned SVD model with SGD. |
| **PMF Model** | `0.8342` | Probabilistic formulation using vectorized batch gradient updates. |
| **Improvement** | **+5.66%** | PMF achieves a noticeably lower error rate by modeling uncertainty and noise. |

---

## 📂 Project Structure

```bash
├── data/                       # Raw MovieLens 1M dataset files (.dat)
├── processed/                  # Generated interaction matrices (git-ignored)
├── models/
│   ├── pmf_model.py            # HighPerformancePMF training and convergence plotting
│   └── svd_model.py            # SVD model training and min-max normalization
├── utils/
│   ├── data_loader.py          # Data ingestion and validation
│   ├── matrix_creation.py      # Train-test splitting and mean-centered normalization
│   └── recommendation.py       # Recommendation engines & real-time Ridge Regression projection
├── notes/                      # Simple, non-tech explanation guides
│   ├── pmf_explanation.md      # PMF model logic & code breakdown
│   └── svd_explanation.md      # SVD model logic & code breakdown
├── reports/                    # Trained model factors, predictions, and metrics (git-ignored)
├── app.py                      # Interactive Streamlit Web App
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 🔧 Setup & Installation

### 1. Ingest Data
Ensure the MovieLens 1M `.dat` files (`movies.dat`, `ratings.dat`, `users.dat`) are placed inside the `data/` folder.

### 2. Configure Environment & Install Dependencies
Create a virtual environment, activate it, and install the locked project dependencies:
```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 3. Preprocess Matrices & Train Models
Run the preprocessing script to create train/test interaction matrices, then train the models to output weights and prediction files:
```bash
# 1. Create matrices
python -m utils.matrix_creation

# 2. Train SVD
python models/svd_model.py

# 3. Train PMF
python models/pmf_model.py
```

### 4. Run the Streamlit Dashboard
Launch the interactive web interface:
```bash
streamlit run app.py
```

---

## 🧠 Core Algorithms Explained

### How We Handle Missing Ratings (Sparsity)
In the MovieLens dataset, users have only rated a tiny fraction of the 3,884 movies, making the user-movie rating grid mostly empty. Both models handle this by **ignoring the blank cells during the training phase** (using the `train_mask` or `np.where(R > 0)` filters). 

They only learn parameters from actual reviews. Once the user profiles ($U$) and movie DNA traits ($V$) are optimized, they are multiplied together ($\text{Prediction} = U \cdot V^T$) to fill all 23.3 million cells of the grid instantly.

### Real-Time Custom User Projection (Ridge Regression)
When a user inputs custom ratings in the **Taste Profiler Sandbox**, the app projects their preferences onto the trained PMF space in real-time. It sets up a regularized **Ridge Regression** system:
$$U_{\text{new}} = (W_{\text{rated}}^T W_{\text{rated}} + \lambda I)^{-1} W_{\text{rated}}^T (R_{\text{rated}} - \mu - b_{\text{items}})$$
This solves for both the user's latent taste coefficients and their bias jointly in milliseconds, providing instant, highly tailored recommendations without needing to re-train the entire model.
