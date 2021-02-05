import tensorflow as tf
import numpy as np
from generic_models import GenericModel, GenericAutoencoder


class DenseAutoencoder(GenericAutoencoder):
    def __init__(self,
                 input_shape,
                 loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(1e-4)):
        encoder = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(int(np.prod(input_shape)/6), activation='relu'),
        ])
        decoder = tf.keras.Sequential([
            tf.keras.layers.Dense(np.prod(input_shape), activation='sigmoid'),
            tf.keras.layers.Reshape(input_shape),
        ])
        super(DenseAutoencoder, self).__init__(encoder, decoder, loss, optimizer)


class ConvAutoencoder(GenericAutoencoder):
    def __init__(self,
                 input_shape,
                 loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(1e-4)):
        if len(input_shape) == 2:
            input_shape = list(input_shape)
            input_shape.append(1)
            input_shape = tuple(input_shape)
        encoder = tf.keras.Sequential([
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=2),
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=2)
        ])
        decoder = tf.keras.Sequential([
            tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', padding='same'),
            tf.keras.layers.Conv2D(3, kernel_size=(3,3), activation='sigmoid', padding='same')
        ])
        super(ConvAutoencoder, self).__init__(encoder, decoder, loss, optimizer)


class UNet(GenericModel):
    def __init__(self,
                 input_shape,
                 loss=tf.keras.losses.MeanSquaredError(),
                 optimizer=tf.keras.optimizers.Adam(1e-4)):
        super(UNet, self).__init__()
        self._loss = loss
        self._optimizer = optimizer
        # contracting path (left side)
        self._d_conv_1 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=1)
        self._d_max_pool_1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self._d_conv_2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=1)
        self._d_max_pool_2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self._d_conv_3 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=1)
        self._d_max_pool_3 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self._d_conv_4 = tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same', strides=1)
        # expanding path (right side)
        self._u_concat = tf.keras.layers.Concatenate(axis=-1)
        self._u_upconv_1 = tf.keras.Sequential([tf.keras.layers.UpSampling2D(size=(2, 2)),
                                                tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=1)])
        self._u_conv_1 = tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same', strides=1)
        self._u_upconv_2 = tf.keras.Sequential([tf.keras.layers.UpSampling2D(size=(2, 2)),
                                                tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=1)])
        self._u_conv_2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=1)
        self._u_upconv_3 = tf.keras.Sequential([tf.keras.layers.UpSampling2D(size=(2, 2)),
                                                tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=1)])
        self._u_conv_3 = tf.keras.layers.Conv2D(16, (3,3), activation='relu', padding='same', strides=1)
        self._u_conv_4 = tf.keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same', strides=1)
        
    def call(self, x):
        # contracting path (left side)
        x_1 = self._d_conv_1(x)
        d_x_1 = self._d_max_pool_1(x_1)
        x_2 = self._d_conv_2(d_x_1)
        d_x_2 = self._d_max_pool_2(x_2)
        x_3 = self._d_conv_3(d_x_2)
        d_x_3 = self._d_max_pool_3(x_3)
        x_4 = self._d_conv_4(d_x_3)
        # expanding path (right side)
        u_y_1 = self._u_upconv_1(x_4)
        y_1 = self._u_conv_1(self._u_concat([x_3, u_y_1]))
        u_y_2 = self._u_upconv_2(y_1)
        y_2 = self._u_conv_2(self._u_concat([x_2, u_y_2]))
        u_y_3 = self._u_upconv_3(y_2)
        y_3 = self._u_conv_3(self._u_concat([x_1, u_y_3]))
        y = self._u_conv_4(y_3)
        return y

class SRCNN(GenericModel):
    def __init__(self,
                 input_shape,
                 loss=tf.keras.losses.MeanSquaredError()):
        super(SRCNN, self).__init__()
        self._loss = loss
        self._optimizer_1 = tf.keras.optimizers.SGD(1e-4)
        self._optimizer_2 = tf.keras.optimizers.SGD(1e-5)
        self._conv_1 = tf.keras.layers.Conv2D(64, (9,9), activation='relu', padding='same', strides=1)
        self._conv_2 = tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', strides=1)
        self._conv_3 = tf.keras.layers.Conv2D(3, (5,5), activation=None, padding='same', strides=1)
        
    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.call(x)
            loss = self._loss(y_hat, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        vars_1 = self.trainable_variables[:4]
        vars_2 = self.trainable_variables[4:]
        gradients_1 = gradients[:4]
        gradients_2 = gradients[4:]
        # Different learning rate depending on layer
        self._optimizer_1.apply_gradients(zip(gradients_1, vars_1))
        self._optimizer_2.apply_gradients(zip(gradients_2, vars_2))
    
    def call(self, x):
        x = self._conv_1(x)
        x = self._conv_2(x)
        x = self._conv_3(x)
        return x