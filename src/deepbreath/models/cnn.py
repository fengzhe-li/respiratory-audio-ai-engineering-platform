from tensorflow import keras

from deepbreath.ml_contract import CLASS_NAMES, CNN_INPUT_SHAPE, NUM_CLASSES

DISEASE_CLASSES = list(CLASS_NAMES)


def build_melspectrogram_cnn(input_shape: tuple[int, int, int] = CNN_INPUT_SHAPE):
    """Build the CNN architecture from the MSc notebook without changing its logic."""

    mspec_input = keras.layers.Input(shape=input_shape, name="mSpecInput")
    x = keras.layers.Conv2D(32, 5, strides=(2, 3), padding="same")(mspec_input)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding="valid")(x)

    x = keras.layers.Conv2D(64, 3, strides=(2, 2), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding="valid")(x)

    x = keras.layers.Conv2D(96, 2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    x = keras.layers.MaxPooling2D(pool_size=2, padding="valid")(x)

    x = keras.layers.Conv2D(128, 2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(keras.activations.relu)(x)
    mspec_output = keras.layers.GlobalMaxPooling2D()(x)

    feature_model = keras.Model(mspec_input, mspec_output, name="mSpecModel")

    input_mspec = keras.layers.Input(shape=input_shape, name="mspec")
    mspec = feature_model(input_mspec)
    concat = keras.layers.concatenate([mspec])
    _ = keras.layers.Dropout(0.2)(concat)
    hidden = keras.layers.Dense(50, activation="relu")(concat)
    hidden = keras.layers.Dropout(0.3)(hidden)
    hidden = keras.layers.Dense(25, activation="relu")(hidden)
    hidden = keras.layers.Dropout(0.3)(hidden)
    output = keras.layers.Dense(NUM_CLASSES, activation="softmax")(hidden)

    return keras.Model([input_mspec], output, name="mSpec_model")


def compile_melspectrogram_cnn(model, learning_rate: float = 0.001):
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=keras.optimizers.Nadam(learning_rate),
        metrics=["accuracy"],
    )
    return model


def default_callbacks():
    return [
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.1,
            patience=3,
            min_lr=0.00001,
            mode="min",
        )
    ]
