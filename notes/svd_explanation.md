# Understanding the SVD (Singular Value Decomposition) Model

Imagine you run a movie streaming service. You have thousands of users and thousands of movies. Users rate movies they watch, but since nobody can watch everything, your rating board is mostly empty. How do you fill in the blanks and recommend a movie a user has never seen?

This is where the **SVD Model** (`models/svd_model.py`) comes in. Here is a simple, non-tech explanation of how it works.

---

## The Analogy: Movie DNA and User Taste Profiles
Instead of grouping movies into traditional categories like "Action" or "Comedy," SVD automatically discovers **hidden themes** (known in tech terms as *latent factors*). 

Imagine each movie has a "DNA profile" made of 20 hidden traits (e.g., how dark it is, how much dialogue it has, how fast-paced it is). 
Likewise, each user has a "Taste profile" showing how much they love or hate each of those 20 traits.

By matching a user's tastes with a movie's DNA, the model can guess the rating.

---

## The Recipe: How Predictions Are Calculated
To predict a rating (e.g., what score Alice would give to *Jurassic Park*), the model combines four ingredients:

1. **Global Average**: The average rating of all movies by all users (e.g., 3.5 stars). This is the starting baseline.
2. **User Bias (The Critic Score)**: Some users are generous (giving 4 or 5 stars to everything), while others are tough critics (giving mostly 1 or 2 stars). The model calculates an adjustment for Alice's rating habits.
3. **Movie Bias (The Popularity Score)**: Some movies are universally loved, and some are widely disliked. The model calculates an adjustment for *Jurassic Park*'s overall quality.
4. **Taste Match (The DNA Dot Product)**: The model multiplies Alice's taste preferences with *Jurassic Park*'s DNA traits. If Alice loves action and *Jurassic Park* has lots of action, this adds bonus points to the prediction.

> **Prediction Formula:**  
> `Predicted Rating = Global Average + Alice's Bias + Jurassic Park's Bias + (Alice's Taste * Jurassic Park's DNA)`

---

## Handling the Blank Spaces (Why Missing Ratings Don't Break It)
In the real world, a user has only rated a tiny fraction of all available movies. The rest of the grid is completely blank. The SVD model handles this with a clever approach:
* **No rating is not a "Zero"**: The model does not assume a missing rating means "0 stars" or that the user hates the movie. It simply treats it as an **unknown**.
* **Learning only from what we know**: In the code, it uses the line `np.where(R > 0)` to locate only the coordinates of ratings that actually exist. During the training phase, the model completely ignores the blank cells and only adjusts its guesses based on actual reviews.
* **Extrapolating to the blanks**: Once the user tastes and movie DNA profiles have been trained using those known ratings, the model multiplies them together for *every single movie and user combination*—even the ones that were blank. This fills in the blanks with predictions.

---

## The Training: How the Model Learns
When we start, the model has no idea what any movie's DNA is or what Alice likes. It starts with random guesses.

1. **Trial and Error**: The model goes through all the actual ratings users have already given.
2. **Calculate the Mistake (Error)**: It guesses a rating, compares it to the real rating, and measures how far off it was.
3. **Small Adjustments (Stochastic Gradient Descent)**: If it guessed too low, it nudges the user tastes and movie DNA values slightly so that next time, the guess will be closer.
4. **Gradual Fine-Tuning**: It does this over and over for 25 rounds (called *epochs*). In each round, it shuffles the ratings to make sure it doesn't get stuck in a pattern. To prevent over-adjusting, the steps it takes get smaller and smaller as it nears the end (known as *learning rate decay*).
5. **Clean Output**: At the very end, it scales all predictions so they fit neatly between 1.0 and 5.0 stars and saves them for the system to use.

---

## Behind the Scenes: A Look at the Code (`svd_model.py`)

If you open the python code file, here is what each major section is doing in plain English:

### 1. The Setup (`train_tuned_biased_svd` function)
```python
def train_tuned_biased_svd(epochs=25, k=20, lr=0.007, reg=0.04):
```
This is the main control room. It takes in a few settings:
* `epochs=25`: Run the learning loop 25 times.
* `k=20`: Find exactly 20 hidden traits (the DNA size) for movies/users.
* `lr=0.007`: The speed of learning (learning rate). Larger means faster but potentially sloppy adjustments; smaller means slower but more precise.
* `reg=0.04`: The safety buffer (regularization) that stops user preferences or movie traits from growing to unrealistic extremes.

### 2. Loading the Data
```python
matrix_df = pd.read_csv("processed/user_item_matrix.csv", index_col=0)
R = matrix_df.values
```
The program reads a spreadsheet (`user_item_matrix.csv`) containing the rating board where rows represent users and columns represent movies. It converts this table into a numerical grid (`R`) that the model can understand.

### 3. Setting Up Blank Sheets
```python
global_mean = np.mean(R[R > 0])
b_u = np.zeros(num_users)
b_i = np.zeros(num_items)
U = np.random.normal(0, 0.1, (num_users, k))
V = np.random.normal(0, 0.1, (num_items, k))
```
The program initializes the parts needed for predictions:
* Calculates the average score (`global_mean`) of all known ratings.
* Creates blank lists for user and movie biases, initially filled with zeros.
* Fills the User (`U`) and Movie (`V`) DNA tables with random, very small numbers as a starting guess.

### 4. The Learning Loop
```python
for epoch in range(epochs):
    np.random.shuffle(samples)
    for u, i, r in samples:
```
The code enters a loop to repeat the learning process 25 times. Inside, it shuffles all user ratings randomly (so it doesn't get biased by learning in the same order) and goes through them one by one.

### 5. Correcting Guesses
```python
pred = global_mean + b_u[u] + b_i[i] + np.dot(U[u], V[i])
err = r - pred

b_u[u] += lr * (err - reg * b_u[u])
b_i[i] += lr * (err - reg * b_i[i])
...
U[u] += lr * (err * V[i] - reg * U[u])
V[i] += lr * (err * u_old - reg * V[i])
```
For each rating:
* It makes a prediction.
* It finds the difference (`err`) between the real rating and its prediction.
* It updates the user bias, movie bias, user tastes (`U`), and movie traits (`V`) by multiplying the error and the learning rate, subtracting a small penalty (`reg`) to keep them stable.

### 6. Smoothing and Normalizing Ratings
```python
scaled_matrix = 1.0 + 4.0 * (preds_all - min_pred) / (max_pred - min_pred)
final_svd_matrix = np.clip(scaled_matrix, 1.0, 5.0)
```
After learning is complete, some predicted ratings might be outside the standard 1-to-5 star range (e.g. 0.8 or 5.2 stars). The code stretches and squeezes the predictions so every single number fits nicely between 1.0 and 5.0 stars.

### 7. Saving the Output
```python
np.save("reports/svd_predictions.npy", final_svd_matrix)
```
Finally, it saves the complete, predicted rating board to a file. The system can now load this file instantly to recommend a movie to any user!

