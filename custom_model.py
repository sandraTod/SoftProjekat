from tensorflow import keras
import os


class CustomCNN:
    def __init__(self, img_size):
        model = keras.Sequential()
        model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(img_size, img_size, 3)))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.2))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(keras.layers.Dense(5, activation='softmax'))
        self.model = model

    def train(self, x, y, batch_size=4, epochs=5):
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        if os.path.exists("custom_model.h5"):
            self.model.load_weights("custom_model.h5")
        else:
            self.model.fit(x, y, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=1)
            self.model.save_weights("custom_model.h5")

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, batch_size=4, verbose=0)
        return accuracy*100
