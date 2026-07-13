import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from utils.data_loader import load_movies
from utils.recommendation import load_pmf_prediction_matrix, load_svd_prediction_matrix

movies_df = load_movies()
num_movies = len(movies_df)
print('movies', num_movies)
svd_matrix = load_svd_prediction_matrix(num_users=6040, num_movies=num_movies)
pmf_matrix = load_pmf_prediction_matrix(num_users=6040, num_movies=num_movies)
print('svd', svd_matrix.shape, 'pmf', pmf_matrix.shape)
common_users = min(svd_matrix.shape[0], pmf_matrix.shape[0])
common_movies = min(svd_matrix.shape[1], pmf_matrix.shape[1])
print('common', common_users, common_movies)
svd_matrix = svd_matrix[:common_users, :common_movies]
pmf_matrix = pmf_matrix[:common_users, :common_movies]
for user_id in [1, 100, 400, 1000, 4000, 6040]:
    idx = user_id - 1
    print('user', user_id)
    u_svd = np.asarray(svd_matrix[idx])
    u_pmf = np.asarray(pmf_matrix[idx])
    print('len', len(u_svd), len(u_pmf), 'nan count', np.isnan(u_svd).sum(), np.isnan(u_pmf).sum())
    try:
        plt.figure()
        plt.scatter(u_svd, u_pmf, alpha=0.2)
        plt.close()
        print('scatter ok')
    except Exception as e:
        print('scatter err', e)
    x = np.linspace(1.0,5.0,400)
    try:
        if len(np.unique(u_svd)) > 1:
            y = gaussian_kde(u_svd)(x)
            print('kde ok', len(y))
        else:
            print('kde could be skipped, unique values', np.unique(u_svd))
    except Exception as e:
        print('kde err', e)
