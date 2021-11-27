import tensorflow as tf
from tensorflow.keras import backend as K


# Implementation of the Rankingloss loss function.
# Based on the implementation of https://github.com/gabzai/Ranking-Loss-SCA, but with small changes for compatibility
# With Tensorflow 2.x.
def dummyloss(y_true, y_pred):
    return 0


def rankingloss(alpha_value=0.5, nb_class=256):
    # Rank loss function
    def ranking_loss_sca(y_true, y_pred):
        alpha = K.constant(alpha_value, dtype='float32')

        # Batch_size initialization
        y_true_int = K.cast(y_true, dtype='int32')
        batch_s = K.cast(K.shape(y_true_int)[0], dtype='int32')

        # Indexing the training set (range_value = (?,))
        range_value = K.arange(0, batch_s, dtype='int64')

        # Get rank and scores associated with the secret key (rank_sk = (?,))
        values_topk_logits, indices_topk_logits = tf.nn.top_k(y_pred, k=nb_class,
                                                              sorted=True)  # values_topk_logits = shape(?, nb_class) ; indices_topk_logits = shape(?, nb_class)

        rank_sk = tf.compat.v1.where(tf.equal(K.cast(indices_topk_logits, dtype='int64'),
                                              tf.reshape(K.argmax(y_true_int),
                                                         [tf.shape(input=K.argmax(y_true_int))[0], 1])))[:,
                  1] + 1  # Index of the correct output among all the hypotheses (shape(?,))
        score_sk = tf.gather_nd(values_topk_logits, K.concatenate(
            [tf.reshape(range_value, [tf.shape(input=values_topk_logits)[0], 1]),
             tf.reshape(rank_sk - 1, [tf.shape(input=rank_sk)[0], 1])]))  # Score of the secret key (shape(?,))

        # Ranking Loss Initialization
        loss_rank = 0

        for i in range(nb_class):
            # Score for each key hypothesis (s_i_shape=(?,))
            s_i = tf.gather_nd(values_topk_logits, K.concatenate(
                [tf.reshape(range_value, [tf.shape(input=values_topk_logits)[0], 1]),
                 i * tf.ones([tf.shape(input=values_topk_logits)[0], 1], dtype='int64')]))

            # Indicator function identifying when (i == secret key)
            indicator_function = tf.ones(batch_s) - (
                    K.cast(K.equal(rank_sk - 1, i), dtype='float32') * tf.ones(batch_s))

            # Logistic loss computation
            logistic_loss = K.log(1 + K.exp(- alpha * (score_sk - s_i))) / K.log(2.0)

            # Ranking Loss computation
            loss_rank = tf.reduce_sum(input_tensor=(indicator_function * logistic_loss)) + loss_rank

        return loss_rank / (K.cast(batch_s, dtype='float32'))

    # Return the ranking loss function
    return ranking_loss_sca
