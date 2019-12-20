from utils import generate_rating_matrix, generate_scores_matrix, recalling, \
                  naive_ranking, ranking, generate_data, reduce_dimension, dfm_ranking
import numpy as np


if __name__ == '__main__':
    rating = np.load('ml-latest-small/rating_matrix.npy')
    scores = np.load('ml-latest-small/scores_matrix.npy')

    user = rating[0]

    print(user)

    candidate = recalling(rating, user=user, k=3, scheme='both')

    print(candidate)

    result = dfm_ranking(rating, scores, user=user, candidate=candidate)
    # result = ranking(rating, scores, user=user, candidate=candidate)
    # result = naive_ranking(scores, user=user, candidate=candidate)

    print(result)

    import pickle
    # read the mapping from movieId to index
    with open('ml-latest-small/movies-to-index.pickle', 'rb') as f:
        ref = pickle.load(f)

    import pandas as pd
    movies = pd.read_csv('ml-latest-small/movies.csv')
    links = pd.read_csv('ml-latest-small/links.csv')

    for i in result:

        print(movies[movies['movieId'] == ref[i]]['title'])

