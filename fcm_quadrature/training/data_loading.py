import tensorflow as tf
import numpy as np


def preprocessData(
    trainingSetPath,
    validSetPath,
    testSetPath=None,
    dtype=np.float32,
    num_outputs=4,
    num_inputs=12,
    verbose=0,
):
    """
    Load and preprocess data for multi-output neural network.

    CSV layout: [num_inputs signed distances | num_outputs quadrature weights]
    Validation and test sets are sorted by the first output column.

    Parameters:
    -----------
    num_outputs : int
        Number of output columns in the CSV files
    num_inputs : int
        Number of input columns (default: 12)

    Returns:
    --------
    data : tuple
        (XTrain, yTrain, XValid, yValid [, XTest, yTest])
    """
    from dask import dataframe

    def _load_csv(path):
        df = dataframe.read_csv(path, header=None, dtype=dtype)
        X = np.array(df.iloc[:, :num_inputs])
        y = np.array(df.iloc[:, -num_outputs:])
        return X, y

    XTrain, yTrain = _load_csv(trainingSetPath)
    XValid, yValid = _load_csv(validSetPath)

    # Sort validation set by first output column (useful for ordered plots)
    sort_idx = np.argsort(yValid[:, 0])
    XValid, yValid = XValid[sort_idx], yValid[sort_idx]

    data = (XTrain, yTrain, XValid, yValid)

    if testSetPath is not None:
        XTest, yTest = _load_csv(testSetPath)
        sort_idx = np.argsort(yTest[:, 0])
        XTest, yTest = XTest[sort_idx], yTest[sort_idx]
        data = (XTrain, yTrain, XValid, yValid, XTest, yTest)

    return data


def buildDataset(
    data,
    batchSizeTrain,
    batchSizeValid,
):
    """Build TensorFlow datasets from preprocessed data"""
    if len(data) == 4:
        XTrain, yTrain, XValid, yValid = data
        testSetExist = False
    elif len(data) == 6:
        testSetExist = True
        XTrain, yTrain, XValid, yValid, XTest, yTest = data

    inputDim = XTrain.shape[-1]
    inputLength = XTrain.shape[0]
    stepPerEpoch = int(np.ceil(inputLength / batchSizeTrain))

    trainSet = tf.data.Dataset.from_tensor_slices((XTrain, yTrain))
    validSet = tf.data.Dataset.from_tensor_slices((XValid, yValid))
    if testSetExist:
        testSet = tf.data.Dataset.from_tensor_slices((XTest, yTest))

    trainSetBatch = trainSet.shuffle(batchSizeTrain*2).batch(
        batchSizeTrain).cache().prefetch(batchSizeTrain*2)
    validSetBatch = validSet.batch(batchSizeValid).cache().prefetch(batchSizeValid)
    sets = (trainSetBatch, validSetBatch)
    if testSetExist:
        testSetBatch = testSet.batch(batchSizeValid).cache().prefetch(batchSizeValid)
        sets = (trainSetBatch, validSetBatch, testSetBatch)

    return *sets, inputDim, stepPerEpoch


def trainModel(
    model: tf.keras.Sequential,
    trainSet: tf.data.Dataset,
    validSet: tf.data.Dataset,
    numEpochs,
    initalCheckpointLoss=None,
    lossCheckpointPath=None,
    verbose=0,
    callbackVerbose=0,
):
    """Train the model with optional checkpointing"""
    callbackList = []

    if lossCheckpointPath is not None:
        lossCheckpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=lossCheckpointPath,
            save_best_only=True,
            monitor="val_loss",
            verbose=callbackVerbose,
            initial_value_threshold=initalCheckpointLoss,
        )
        callbackList.append(lossCheckpoint)

    history = model.fit(
        trainSet,
        epochs=numEpochs,
        validation_data=validSet,
        verbose=verbose,
        callbacks=callbackList
    )

    return history


def saveHistory(history, historyPath):
    """Save training history to pickle file"""
    import pickle, os
    directory = os.path.dirname(historyPath)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(historyPath, 'wb') as file:
        pickle.dump(history, file)


def loadHistory(historyPath):
    """Load training history from pickle file"""
    import pickle
    with open(historyPath, 'rb') as file:
        history = pickle.load(file)
    return history
