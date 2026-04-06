import tensorflow as tf


def buildSequentialModel(
    inputDim,
    dtype,
    layerSizeList,
    activationList,
    weightInitializer,
    modelName,
    weightRegularizer=None,
    lossFunc='mse',
    num_outputs=4,
    learningRate=None,
    metricFunc=None,
    verbose=0,
    inputNormLayer=None,
    dropout_rate=0.0,
):
    """
    Build a sequential neural network model with multiple outputs.

    Parameters:
    -----------
    num_outputs : int
        Number of output neurons (default: 4 for quadrature weights)
    dropout_rate : float
        Dropout rate after each hidden layer (default: 0.0 = no dropout)
    """
    from tensorflow import keras
    from keras import layers

    model = keras.Sequential(name=modelName)

    # Ensure inputDim is a tuple for compatibility with Keras 3.x
    if isinstance(inputDim, int):
        input_shape = (inputDim,)
    else:
        input_shape = inputDim
    model.add(keras.Input(shape=input_shape, dtype=dtype))

    if inputNormLayer is not None:
        model.add(inputNormLayer)

    for idxLayer, (layerSize, activation) in enumerate(zip(layerSizeList, activationList)):
        model.add(
            layers.Dense(
                layerSize,
                activation=activation,
                kernel_initializer=weightInitializer,
                bias_initializer='zeros',
                kernel_regularizer=weightRegularizer,
                name=f'dense{idxLayer}'
            )
        )
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate, name=f'dropout{idxLayer}'))

    # Output layer with num_outputs neurons
    model.add(
        layers.Dense(
            num_outputs,
            activation=None,
            kernel_initializer=weightInitializer,
            bias_initializer='zeros',
            kernel_regularizer=weightRegularizer,
            name="outputLayer"
        )
    )

    if verbose > 0:
        model.summary()

    if learningRate is not None:
        optimizer = keras.optimizers.Adam(learning_rate=learningRate)
    else:
        optimizer = keras.optimizers.Adam()

    model.compile(
        optimizer=optimizer,
        loss=lossFunc,
        metrics=metricFunc,
    )
    return model
