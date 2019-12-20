import numpy as np
import pandas as pd
import tensorflow as tf

import pickle
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from queue import PriorityQueue


"""                                   FUNCTIONS for BUILDING NETWORK                                   """

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def cross_entropy(label, prop):
    return -(label * tf.log(tf.clip_by_value(prop, 1e-10, 1.)) +
             (1 - label) * tf.log(tf.clip_by_value(1. - prop, 1e-10, 1.)))


def batch_norm(x, axis=-1, training=True):
    return tf.layers.batch_normalization(
        inputs=x, axis=axis,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=training, fused=True)


def fc_layer(x, units, training=True, dropout=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        inputs = tf.layers.dense(x, units=units, activation=None, use_bias=False)
        inputs = tf.nn.relu(batch_norm(inputs, training=training))
        if dropout:
            return tf.layers.dropout(inputs, rate=0.25, training=training, name='output')
        else:
            return inputs


def bilinear_layer(x, units, training=True, name=''):
    with tf.variable_scope(name_or_scope=name):
        shortcut = x
        inputs = fc_layer(x, units, training=training, name='fc_0')
        inputs = fc_layer(inputs, units, training=training, name='fc_1')
        return tf.add_n([inputs, shortcut], name='output')


"""                                   FUNCTIONS for RECALLING MODULE                                   """

num_movie = 8728
num_user = 610
num_feature = 1128


def generate_rating_matrix(file, dump=False):
    # read raw data
    raw = pd.read_csv(file)
    # read the mapping from movieId to index
    with open('ml-latest-small/movies-to-index.pickle', 'rb') as f:
        ref = pickle.load(f)
        ref = dict(zip(ref.values(), ref.keys()))

    # create a matrix with a shape of (num_user, num_movie)
    rating = np.zeros(shape=(num_user, num_movie))
    # generate rating matrix
    for _, row in raw.iterrows():
        if _ % 10000 == 0:
            print('processing: %d' % _)
        rating[int(row['userId'] - 1), int(ref[row['movieId']])] = row['rating']

    # dump matrix to file
    if dump:
        np.save('rating_matrix.npy', rating)
    # return
    return rating


def user_collaborative_filtering(rating, user, k):
    # define a priority queue for collaborative filtering
    heap = PriorityQueue()
    # perform user-based collaborative filtering
    for i in range(num_user):
        # calculate L2 distance between 'user' and 'i'
        node = (-np.sum(np.square(user - rating[i])), i)
        # put the new node onto heap
        heap.put(node)
        # preserve only the top-k smallest items
        while heap.qsize() > k:
            heap.get()

    # retrieve the top-k smallest items
    result = []
    while not heap.empty():
        # get the index of a given item
        _, index = heap.get()
        # push it into 'result'
        result.append(index)

    # return
    return result


def item_collaborative_filtering(rating, item, k):
    # transpose the rating matrix
    rating = rating.T

    # define a priority queue for collaborative filtering
    heap = PriorityQueue()

    # perform item-based collaborative filtering
    for i in range(num_movie):
        # calculate L2 distance between 'user' and 'i'
        node = (-np.sum(np.square(rating[item] - rating[i])), i)
        # put the new node onto heap
        heap.put(node)
        # preserve only the top-k smallest items
        while heap.qsize() > k:
            heap.get()

    # retrieve the top-k smallest items
    result = []
    while not heap.empty():
        # get the index of a given item
        _, index = heap.get()
        # push it into 'result'
        result.append(index)

    # return
    return result


def recalling(rating, user, k, scheme='user'):
    # define a set for storing result of recalling
    result = set()

    if scheme == 'user':
        # perform user-based collaborative filtering
        neighbors = user_collaborative_filtering(rating, user, k)
        for i in range(num_movie):
            for n in neighbors:
                if rating[n, i] and not user[i]:
                    result.add(i)
    elif scheme == 'item':
        # perform item-based collaborative filtering
        for i in range(num_movie):
            if user[i]:
                result.update(item_collaborative_filtering(rating, i, k))
        for i in range(num_movie):
            if user[i]:
                result.remove(i)
    elif scheme == 'both':
        # perform item-based collaborative filtering
        item_result = set()
        for i in range(num_movie):
            if user[i]:
                item_result.update(item_collaborative_filtering(rating, i, k))
        for i in range(num_movie):
            if user[i]:
                item_result.remove(i)
        # perform user-based collaborative filtering
        user_result = set()
        neighbors = user_collaborative_filtering(rating, user, k)
        for i in range(num_movie):
            for n in neighbors:
                if rating[n, i] and not user[i]:
                    user_result.add(i)
        # obtain the intersection of item result and user result
        result = item_result.intersection(user_result)

    # return
    return result


"""                                   FUNCTIONS for RANKING MODULE                                   """


def standardize(user_features, movie_features):
    # not implemented yet
    return user_features, movie_features


def slice_data(user_features, movie_features, y):
    return train_test_split(np.column_stack((user_features, movie_features)), y,
                            test_size=0.2, random_state=0)


def reduce_dimension(data):
    try:
        # attempt to load a PCA model from disk
        with open('ml-latest-small/pca.pickle', 'rb') as f:
            pca = pickle.load(f)
    except FileNotFoundError:
        # generate a new PCA model
        pca = PCA(n_components=int(num_feature/2), svd_solver='auto')
        # fit the model with 'data'
        pca.fit(data)
        # save the model
        with open('ml-latest-small/pca.pickle', 'wb') as f:
            pickle.dump(pca, f)
    # return data after dimensionality reduction
    return pca.transform(data.reshape(-1, num_movie))


def generate_scores_matrix(file, dump=False):
    # read raw data
    raw = pd.read_csv(file)
    # read the mapping from movieId to index
    with open('ml-latest-small/movies-to-index.pickle', 'rb') as f:
        ref = pickle.load(f)
        ref = dict(zip(ref.values(), ref.keys()))

    # create a matrix with a shape of (num_movie, num_feature)
    scores = np.zeros(shape=(num_movie, num_feature))
    # generate scores matrix
    for _, row in raw.iterrows():
        if _ % 10000 == 0:
            print('processing: %d' % _)
        scores[int(ref[row['movieId']]), int(row['tagId'] - 1)] = row['relevance']

    # dump matrix to file
    if dump:
        np.save('scores_matrix.npy', scores)
    # return
    return scores


def generate_data(rating_matrix):
    try:
        return np.load('ml-latest-small/user_features.npy'), \
               np.load('ml-latest-small/movie_features.npy'), \
               np.load('ml-latest-small/ground_truth.npy')
    except FileNotFoundError:
        user_features, movie_features, ground_truth = [], [], []

        for user in rating_matrix:
            for i in range(num_movie):
                if user[i]:
                    # leave one out
                    loo = np.copy(user)
                    loo[i] = 0
                    # append
                    user_features.append(reduce_dimension(loo).reshape(-1, ))
                    movie_features.append(i)
                    ground_truth.append(user[i])

        return np.array(user_features), np.array(movie_features), np.array(ground_truth)


def naive_ranking(scores, user, candidate):
    # calculate the center of watched movies of a given user
    center = np.zeros(shape=(num_feature, ))
    # count each watched movie
    count = 0
    for i in range(num_movie):
        if user[i]:
            center += scores[i]
            count += 1
    # obtain the mean value of watched movies
    center = center / count
    # calculate the distance from the center for each candidate
    distance = np.array([np.sum(np.square(center - scores[m])) for m in candidate])
    # sort the candidate by their distances from center, then return
    return np.array(list(candidate))[distance.argsort()]


def ranking(rating, scores, user, candidate):
    from users.rating.recall import RecallingModule

    try:
        return RecallingModule.predict(scores, user, candidate)
    except FileNotFoundError:
        model = RecallingModule(rating, scores)
        model.train()

        return RecallingModule.predict(scores, user, candidate)


def dfm_ranking(rating, scores, user, candidate):
    from sklearn.metrics import mean_squared_error
    from users.rating.DeepFM import DeepFM

    # params
    dfm_params = {
        "use_fm": True,
        "use_deep": True,
        "embedding_size": 8,
        "dropout_fm": [1.0, 1.0],
        "deep_layers": [128, 64],
        "dropout_deep": [0.5, 0.5, 0.5],
        "deep_layers_activation": tf.nn.relu,
        "epoch": 30,
        "batch_size": 256,
        "learning_rate": 0.001,
        "optimizer_type": "adam",
        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        "l2_reg": 0.01,
        "verbose": True,
        "loss_type": "mse",
        "eval_metric": mean_squared_error,
        "random_seed": 2017
    }

    dfm_params["feature_size"] = int(1.5 * num_feature)
    dfm_params["field_size"] = int(1.5 * num_feature)

    # get the number of candidates
    num_candidate = len(candidate)
    candidate = np.array(list(candidate), dtype=int)
    # perform PCA and repeat 'user'
    user = np.repeat(reduce_dimension(user).reshape(1, -1), num_candidate, axis=0).reshape(-1, int(num_feature / 2))
    # get movie features
    item = scores[candidate]
    # prepare testing data in the required format
    Xv_test = np.concatenate((user, item), axis=1)
    Xi_test = np.repeat(np.array(range(Xv_test.shape[1])).reshape(1, -1), Xv_test.shape[0], axis=0) \
                .reshape(-1, Xv_test.shape[1])

    try:
        graph = tf.get_default_graph()
        # session
        with tf.Session(graph=graph) as sess:
            loader = tf.train.import_meta_graph('dfm/model.meta')

            # get input tensor
            feat_index = graph.get_tensor_by_name('feat_index:0')
            feat_value = graph.get_tensor_by_name('feat_value:0')
            dropout_keep_fm = graph.get_tensor_by_name('dropout_keep_fm:0')
            dropout_keep_deep = graph.get_tensor_by_name('dropout_keep_deep:0')
            train_phase = graph.get_tensor_by_name('train_phase:0')

            # get output tensor
            out = graph.get_tensor_by_name('out:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('dfm'))

            pred = sess.run(out, feed_dict={feat_index: Xi_test,
                                            feat_value: Xv_test,
                                            dropout_keep_fm: [1.0] * len(dfm_params['dropout_fm']),
                                            dropout_keep_deep: [1.0] * len(dfm_params['dropout_deep']),
                                            train_phase: False})

            return candidate[pred.reshape(-1, ).argsort()][::-1]

    except FileNotFoundError:
        # init a DeepFM model
        dfm = DeepFM(**dfm_params)

        # prepare training and validation data in the required format
        user_features, movie_features, y = generate_data(reduce_dimension(rating))
        Xv_train, Xv_valid, y_train, y_valid = slice_data(user_features, movie_features, y)

        # fit a DeepFM model
        dfm.fit(scores, Xv_train, y_train, Xv_valid, y_valid, early_stopping=True, refit=True)

        # make prediction
        return candidate[dfm.predict(scores, Xv_test).reshape(-1, ).argsort()][::-1]
