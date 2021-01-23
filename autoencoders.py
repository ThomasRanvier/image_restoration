import tensorflow as tf
from generic_model import GenericAutoencoder

class Autoencoder(GenericAutoencoder):
    def __init__(self,
                 encoder,
                 decoder,
                 loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(1e-4)):
        super(Autoencoder, self).__init__(loss, optimizer, encoder, decoder)