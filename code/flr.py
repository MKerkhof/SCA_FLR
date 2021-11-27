import focal_loss as flc
import tensorflow as tf


# Focal loss ratio,
def flr_loss(n, num_classes=9, alpha=None, gamma=2.0):
    fl = flc.categorical_focal_loss(alpha=alpha
                                    , gamma=gamma)

    def flr(y_true, y_pred):
        ce = fl(y_true, y_pred)

        ce_shuffled = 0.0

        for i in range(n):
            y_true_shuffled = tf.random.shuffle(y_true)
            ce_shuffled += fl(y_true_shuffled, y_pred)

        ce_shuffled = ce_shuffled / n

        return ce / ce_shuffled

    return flr
