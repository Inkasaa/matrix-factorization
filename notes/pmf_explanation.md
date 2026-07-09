# Understanding the PMF (Probabilistic Matrix Factorization) Model

The **PMF Model** (`models/pmf_model.py`) is a cousin of the SVD model. It is designed to solve the same problem—predicting what rating a user would give to a movie they haven't seen yet—but it goes about it in a faster, more mathematically structured way.

Here is a simple, non-tech breakdown of the PMF model.

---

## What makes PMF different from SVD?

If SVD is like a craftsman carefully adjusting one rating at a time, PMF is like an automated factory line updating thousands of ratings at once.

While both models break down the user-movie rating table into hidden user tastes and movie traits (latent factors), PMF uses probability theory to handle the "noise" and uncertainty of real-world rating data.

---

## Handling the Blank Spaces (Why Missing Ratings Don't Break It)
Similar to SVD, PMF solves the challenge where most cells in the user-movie rating grid are blank:
* **Selective Training Mask**: In the code, the model uses a structure called `train_mask`. This is a giant checklist of `True` and `False` values indicating which cells contain a real rating.
* **Skipping the Blanks**: The line `np.where(train_mask)` tells the model to only pull out coordinates of actual reviews. When the model calculates predicted values and errors, it does so *only* for these existing reviews, ignoring the empty gaps.
* **Creating Predictions for All**: Once the user traits (`U`) and movie traits (`V`) are optimized using the known ratings, the code calculates the final predictions for every single possible user-movie combination in parallel (in the `predict_all` method). This instantly fills in all the blank spaces with predicted ratings.

---

## Key Features of the PMF Code

### 1. Vector-Gradient Optimization (Batch Updates)
Instead of looking at ratings one by one, the PMF model processes updates in massive chunks (50,000 ratings at a time). 
* **Why this matters**: Processing in batches is much faster and uses modern computer chips more efficiently. This allows the model to scale to millions of ratings without slowing down.

### 2. Regularization (Avoiding "Over-fitting")
The model includes a safety mechanism (a parameter called `reg`). It penalizes the model if user taste scores or movie DNA traits become too extreme.
* **Why this matters**: This prevents the model from memorizing the training data too perfectly. By keeping the values reasonable, the model remains good at predicting ratings for *new* movies that the user hasn't rated yet.

### 3. Visual Convergence Curves
Every time the PMF model is trained, it automatically saves a graph (`reports/pmf_convergence.png`). 
* **Why this matters**: This chart visualizes how the model’s error drops round by round (epoch by epoch). If the line flattens out, it means the model has finished learning and has successfully "converged" to its best settings.

### 4. Output Storage for Dashboards
Once training finishes, PMF saves its learned user profiles (`U.npy`), movie profiles (`V.npy`), biases (`b_u.npy`, `b_i.npy`), and the global average (`mu.npy`) as files.
* **Why this matters**: The website or dashboard app can load these files instantly to give real-time recommendations to users without needing to re-train the AI from scratch.

---

## Performance Comparison
In this project:
* The SVD model achieves an error rate (RMSE) of **0.8842**.
* The PMF model achieves an error rate (RMSE) of **0.8342**.
* **Result**: PMF performs about **5.66% better** than SVD, meaning its recommendations are noticeably more accurate.

---

## Behind the Scenes: A Look at the Code (`pmf_model.py`)

If you open the python code file, here is what each major section is doing in plain English:

### 1. The Structure (`HighPerformancePMF` Class)
```python
class HighPerformancePMF:
    def __init__(self, n_factors=15, lr=0.0005, reg=0.02, max_iter=15):
```
This defines the PMF model object. When we build the model, we set its tuning knobs:
* `n_factors=15`: It will find 15 hidden characteristics (latent factors) for movies and users.
* `lr=0.0005`: The learning rate (speed).
* `reg=0.02`: The safety penalty to prevent the model from over-tuning.
* `max_iter=15`: The model will run its training loops 15 times.

### 2. Preparing the Tables (`fit` method)
```python
def fit(self, R, train_mask):
    n_users, n_items = R.shape
    self.mu = float(np.mean(R[train_mask]))
    self.b_u = np.zeros(n_users)
    self.b_i = np.zeros(n_items)
    self.U = np.random.normal(0, 0.05, (n_users, self.n_factors))
    self.V = np.random.normal(0, 0.05, (n_items, self.n_factors))
```
This function is where the learning happens:
* It sets up the global rating average (`self.mu`).
* It sets user biases (`b_u`) and item biases (`b_i`) to all zeroes.
* It fills the user preference matrix `U` and movie trait matrix `V` with random, very small numbers to begin the guessing game.

### 3. The Batch Learning Loop
```python
for iteration in range(self.max_iter):
    # Predict all current ratings
    preds = self.mu + self.b_u[users_idx] + self.b_i[items_idx] + np.sum(self.U[users_idx] * self.V[items_idx], axis=1)
    errors = ratings - preds
```
In each of the 15 training rounds:
* It predicts the score for all ratings simultaneously.
* It calculates the difference (`errors`) between what the user actually rated and what the model guessed.

### 4. Updating Parameters in Chunks
```python
for idx in range(0, len(errors), 50000):
    batch_slice = slice(idx, idx + 50000)
    # ... update biases and latent factor matrices in slices of 50,000 ratings ...
```
To keep the computer running fast and prevent memory issues, it splits the ratings into chunks of 50,000. It updates all user/movie traits and biases in these chunks using fast matrix math.

### 5. Running the Pipeline (`run_pmf_pipeline` function)
```python
def run_pmf_pipeline():
    matrices = create_and_process_matrices()
    pmf = HighPerformancePMF(max_iter=15)
    pmf.fit(R_train, train_mask)
```
This is the coordinator function that executes when you run the script:
* **Loads data**: Calls utility functions to prepare the rating matrices.
* **Trains PMF**: Builds a PMF model and runs the learning process (`fit`).

### 6. Plotting and Saving Results
```python
plt.savefig("reports/pmf_convergence.png", dpi=150)
np.save("reports/pmf_factors/U.npy", pmf.U)
# ... save other factors and update model_metrics.json ...
```
* **Saves a plot**: Creates a chart showing how the error drops with each iteration.
* **Saves factor matrices**: Saves the mathematical arrays so they can be loaded instantly by a web app or dashboard without training again.
* **Updates metrics**: Records the PMF accuracy score inside the overall project metrics file (`reports/model_metrics.json`).

