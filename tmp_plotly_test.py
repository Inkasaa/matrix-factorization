import numpy as np
from utils.data_loader import load_movies
from utils.recommendation import load_pmf_prediction_matrix, load_svd_prediction_matrix
from scipy.stats import gaussian_kde
import plotly.graph_objects as go

movies_df = load_movies()
num_movies = len(movies_df)
print('movies', num_movies)
svd_matrix = load_svd_prediction_matrix(num_users=6040, num_movies=num_movies)
pmf_matrix = load_pmf_prediction_matrix(num_users=6040, num_movies=num_movies)
print('svd', svd_matrix.shape, 'pmf', pmf_matrix.shape)
common_movies = min(svd_matrix.shape[1], pmf_matrix.shape[1])
svd_matrix = svd_matrix[:, :common_movies]
pmf_matrix = pmf_matrix[:, :common_movies]
for user_id in [1, 100, 400, 1000, 4000, 6040]:
    u_svd = np.asarray(svd_matrix[user_id-1])
    u_pmf = np.asarray(pmf_matrix[user_id-1])
    print('user', user_id, len(u_svd), len(u_pmf))
    x = np.linspace(1.0, 5.0, 400)
    if len(np.unique(u_svd)) > 1:
        kde = gaussian_kde(u_svd)
        y = kde(x)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, fill='tozeroy'))
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=u_svd, y=u_pmf, mode='markers', marker=dict(opacity=0.25)))
print('done')
