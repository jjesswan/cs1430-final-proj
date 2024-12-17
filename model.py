import numpy as np
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense

class CNNModel(tf.keras.Model):
    def __init__(self, num_classes, learning_rate):
        super(CNNModel, self).__init__()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        # Define layers explicitly
        self.conv1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu')
        self.pool1 = MaxPool2D(pool_size=(2, 2))
        self.conv2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')
        self.pool2 = MaxPool2D(pool_size=(2, 2))
        self.conv3 = Conv2D(filters=128, kernel_size=(3, 3), activation='relu')
        self.pool3 = MaxPool2D(pool_size=(2, 2))
        self.flatten = Flatten()
        self.dense1 = Dense(units=64, activation='relu')
        self.output_layer = Dense(units=num_classes, activation='softmax')

    def call(self, inputs, return_features=False):
        """ Pass input through network. """
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)

        #  Extract features before the dense layers
        features = self.flatten(x)  
        x = self.dense1(features)
        outputs = self.output_layer(x)

        # Optionally return features
        if return_features:
            return features
        return outputs

    @staticmethod
    def loss_fn(labels, predictions):
        """ Loss function for the model. """
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        return loss(labels, predictions)


def getModel(train_data, train_labels, test_data, test_labels, num_classes, img_size, learning_rate=0.001, batch_size=32, epochs=10):
    model = CNNModel(num_classes, learning_rate)

    # Build model by passing a sample input
    sample_input = tf.keras.Input(shape=(img_size, img_size, 3))
    model.build((None, img_size, img_size, 3))
    # Explicity build model graph
    model.call(sample_input) 
    model.summary()

    # Feature extraction
    feature_extractor = tf.keras.Model(
        inputs=sample_input,
        outputs=model.call(sample_input, return_features=True)
    )
    feature_extractor.summary()

    # Compile the model
    model.compile(
        optimizer=model.optimizer,
        loss=model.loss_fn,
        metrics=["sparse_categorical_accuracy"]
    )

    # Train model
    model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs)

    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_data, test_labels, batch_size=batch_size)
    print("Validation Loss:", test_loss)
    print("Validation Accuracy:", test_accuracy)

    return model, feature_extractor
