import keras.backend as K

def boosted_loss(y_true, y_pred):
    when_i_dont_care = K.zeros(y_true.shape)
    when_i_do_care = K.log(y_true) - K.log(y_pred)
    return K.sum(K.where(y_true != y_pred, when_i_dont_care, when_i_do_care))
