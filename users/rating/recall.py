import os
import time
import numpy as np
import tensorflow as tf

from users.rating.utils import num_feature, generate_data, standardize, slice_data, reduce_dimension
from users.rating.utils import fc_layer, bilinear_layer


class RecallingModule:
    def __init__(self,
                 rating_matrix,
                 scores_matrix,
                 batch_size=256,
                 num_epochs=None,
                 iterations=1e5,
                 learning_rate=5e-4):
        # original data
        self.rating_matrix = rating_matrix
        self.scores_matrix = scores_matrix
        # hyper-parameters
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.iterations = int(iterations)
        self.learning_rate = learning_rate
        # initialization of tensors
        with tf.variable_scope(name_or_scope='init'):
            # indicator of training
            self.training = tf.placeholder(tf.bool, name='training')
            # input tensors
            self._user = tf.placeholder(tf.float32, [None, num_feature/2], name='user')
            self._movie = tf.placeholder(tf.float32, [None, num_feature], name='movie')
            # ground truth
            self._rating = tf.placeholder(tf.float32, [None, 1], name='rating')

    def model(self):
        with tf.variable_scope(name_or_scope='recalling_module'):
            # pre-processing layers
            user_fc_0 = fc_layer(self._user, 256, training=self.training, name='user_fc_0')
            movie_fc_0 = fc_layer(self._movie, 256, training=self.training, name='movie_fc_0')
            # bi-linear layers
            user_bi_0 = bilinear_layer(user_fc_0, 256, training=self.training, name='user_bi_0')
            movie_bi_0 = bilinear_layer(movie_fc_0, 256, training=self.training, name='movie_bi_0')
            # post-processing layers
            user_fc_1 = fc_layer(user_bi_0, 128, training=self.training, name='user_fc_1')
            movie_fc_1 = fc_layer(movie_bi_0, 128, training=self.training, name='movie_fc_1')
            # merge user features and movie features by an add_n operation, following by a fully-connected layer
            # merge = fc_layer(tf.add_n([user_fc_1, movie_fc_1]), 64, training=self.training, name='merge')
            merge = fc_layer(tf.multiply(user_fc_1, movie_fc_1), 64, training=self.training, name='merge')
            # output layer
            rating_output = tf.layers.dense(merge, units=1, activation=None, use_bias=False, name='rating_output')
            # return
            return rating_output

    def train(self):
        # generate data from rating matrix and scores matrix
        user_features, movie_features, y = generate_data(self.rating_matrix)

        # perform standardization
        user_features, movie_features = standardize(user_features, movie_features)

        # slice data
        x_train, x_test, y_train, y_test = slice_data(user_features, movie_features, y)
        print('random loss = %f' % np.mean(np.square(5 * np.random.rand(y_test.shape[0]) - y_test)))

        # generate a new network
        rating = self.model()
        rating_summary = tf.summary.histogram('rating_summary', rating)

        with tf.variable_scope(name_or_scope='optimizer'):
            # loss function
            total_loss = tf.reduce_mean(tf.square(self._rating - rating), name='total_loss')
            loss_summary = tf.summary.scalar('loss_summary', total_loss)

            # optimizer
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_ops = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(total_loss)

        # configuration
        if not os.path.isdir('save'):
            os.mkdir('save')
        config = tf.ConfigProto()

        print('Start training')
        with tf.Session(config=config) as sess:
            # initialization
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

            # saver
            optimal = np.inf
            saver = tf.train.Saver(max_to_keep=5)

            # store the network graph for tensorboard visualization
            writer = tf.summary.FileWriter('save/network_graph', sess.graph)
            merge_op = tf.summary.merge([rating_summary, loss_summary])

            # data set
            queue = tf.train.slice_input_producer([x_train, y_train],
                                                  num_epochs=self.num_epochs, shuffle=True)
            x_batch, y_batch = tf.train.batch(queue, batch_size=self.batch_size, num_threads=1,
                                              allow_smaller_final_batch=False)

            # enable coordinator
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord)

            try:
                for i in range(self.iterations):
                    # retrieve mini-batch
                    x, y = sess.run([x_batch, y_batch])

                    # update parameters
                    _, loss, sm = sess.run([train_ops, total_loss, merge_op],
                                           feed_dict={self.training: True,
                                                      self._user: x[:, :-1],
                                                      self._movie: self.scores_matrix[x[:, -1].astype(int)],
                                                      self._rating: y.reshape(-1, 1)})

                    # examine result
                    if i % 100 == 0:
                        print('iteration %d: loss = %f' % (i, loss))
                        writer.add_summary(sm, i)
                        writer.flush()
                    if i % 500 == 0:
                        prob, loss = sess.run([rating, total_loss],
                                              feed_dict={self.training: False,
                                                         self._user: x_test[:, :-1],
                                                         self._movie: self.scores_matrix[x_test[:, -1].astype(int)],
                                                         self._rating: y_test.reshape(-1, 1)})
                        # save the optimal models
                        if loss < optimal:
                            optimal = loss
                            print('save at iteration %d with optimal loss of %f' % (i, optimal))
                            saver.save(sess, 'save/%s/model' %
                                       (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
                saver.save(sess, 'save/%s/model' %
                           (time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(time.time()))))
                writer.close()

            finally:
                coord.request_stop()

            coord.join(threads)

    @staticmethod
    def predict(scores, user, candidate):
        # get the number of candidates
        num_candidate = len(candidate)
        candidate = np.array(list(candidate), dtype=int)

        # perform PCA and repeat 'user'
        user = np.repeat(reduce_dimension(user).reshape(1, -1), num_candidate, axis=0).reshape(-1, int(num_feature/2))

        # get movie features
        item = scores[candidate]

        # perform standardization
        user, item = standardize(user, item)

        # get graph
        graph = tf.get_default_graph()
        # session
        with tf.Session(graph=graph) as sess:
            # restore the latest model
            file_list = os.listdir('save/')
            file_list.sort(key=lambda val: val)
            loader = tf.train.import_meta_graph('save/%s/model.meta' % file_list[-2])

            # get input tensor
            training_tensor = graph.get_tensor_by_name('init/training:0')
            user_tensor = graph.get_tensor_by_name('init/user:0')
            movie_tensor = graph.get_tensor_by_name('init/movie:0')

            # get output tensor
            output_tensor = graph.get_tensor_by_name('recalling_module/rating_output/MatMul:0')

            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
            loader.restore(sess, tf.train.latest_checkpoint('save/%s' % file_list[-2]))

            rating = sess.run(output_tensor,
                              feed_dict={training_tensor: False, user_tensor: user, movie_tensor: item})

            return candidate[rating.reshape(-1, ).argsort()][::-1]
