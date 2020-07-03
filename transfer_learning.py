from tensorflow import keras
import os


class VGGBasedModel:
    def __init__(self, img_size):
        self.model_path = "vgg_model.h5"
        base_model = keras.applications.VGG16(include_top=False, input_shape=(img_size, img_size, 3))
        base_model.trainable = False
        inputs = keras.Input(shape=(img_size, img_size, 3))
        x = base_model(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(5, activation='softmax')(x)
        self.model = keras.Model(inputs=inputs, outputs=outputs)

    def train(self, x, y, batch_size=2, epochs=5):
        self.model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=['accuracy'])
        if os.path.exists(self.model_path):
            self.model.load_weights(self.model_path)
        else:
            self.model.fit(x, y, validation_split=0.2, batch_size=batch_size, epochs=epochs, verbose=1)
            self.model.save_weights(self.model_path)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test, y_test, batch_size=4, verbose=0)
        return accuracy*100
