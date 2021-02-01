import tensorflow as tf


class GenericModel(tf.keras.Model):
    def __init__(self):
        super(GenericModel, self).__init__()
    
    @tf.function
    def _train_step(self, x, y):
        with tf.GradientTape() as tape:
            y_hat = self.call(x)
            loss = self._loss(y_hat, y)
        gradients = tape.gradient(loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
    def _run_one_epoch(self, train_dataset):
        for _, (x, y) in enumerate(train_dataset):
            self._train_step(x, y)
    
    def fit(self, train_dataset, eval_dataset, epochs=10):
        print('training start')
        eval_losses = []
        for epoch in range(epochs):
            self._run_one_epoch(train_dataset)
            eval_loss, _ = self.evaluate(eval_dataset)
            eval_losses.append(eval_loss)
            print(f'epoch {epoch + 1}/{epochs} - mean loss {round(eval_loss, 4)}')
        print('training stop')
        return eval_losses
        
    def call(self, x):
        raise NotImplementedError('subclasses must override the call method!')
    
    def evaluate(self, eval_dataset):
        total_loss = 0.
        for i, (x, y) in enumerate(eval_dataset):
            loss_value, y_hat = self._eval_step(x, y)
            total_loss += loss_value
        return (total_loss / (i + 1), (x.numpy(), y.numpy(), y_hat.numpy()))
        
    def _eval_step(self, x, y):
        y_hat = self.call(x)
        return self._loss(y_hat, y).numpy(), y_hat
        

class GenericAutoencoder(GenericModel):
    def __init__(self,
                 encoder,
                 decoder,
                 loss,
                 optimizer):
        super(GenericAutoencoder, self).__init__()
        self._loss = loss
        self._optimizer = optimizer
        self._encoder = encoder
        self._decoder = decoder
    
    def encode(self, x):
        return self._encoder(x)
    
    def decode(self, x):
        return self._decoder(x)
    
    def call(self, x):
        return self.decode(self.encode(x))