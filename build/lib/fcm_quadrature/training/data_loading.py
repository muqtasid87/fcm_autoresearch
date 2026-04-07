import tensorflow as tf
import numpy as np


def _load_csv_single(path, num_inputs, num_outputs, dtype):
    """Load a single CSV file and split into X, y."""
    from dask import dataframe
    df = dataframe.read_csv(path, header=None, dtype=dtype)
    X = np.array(df.iloc[:, :num_inputs])
    y = np.array(df.iloc[:, -num_outputs:])
    return X, y


def _load_csv_multi(paths, num_inputs, num_outputs, dtype):
    """Load and concatenate multiple CSV files."""
    all_X, all_y = [], []
    for path in paths:
        X, y = _load_csv_single(path, num_inputs, num_outputs, dtype)
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def auto_detect_num_inputs(path, num_outputs=4):
    """Detect num_inputs from a metadata JSON or CSV column count.

    Looks for <csv_path>_metadata.json first. Falls back to
    counting CSV columns minus num_outputs.
    """
    import os, json
    meta_path = path.replace('.csv', '_metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        return meta.get('num_inputs', 12)

    # Fallback: read first line of CSV
    with open(path) as f:
        first_line = f.readline().strip()
    n_cols = len(first_line.split(','))
    return n_cols - num_outputs


def preprocessData(
    trainingSetPath,
    validSetPath,
    testSetPath=None,
    dtype=np.float32,
    num_outputs=4,
    num_inputs=None,
    verbose=0,
):
    """
    Load and preprocess data for multi-output neural network.

    CSV layout: [num_inputs signed distances | num_outputs quadrature weights]
    Validation and test sets are sorted by the first output column.

    Parameters:
    -----------
    trainingSetPath : str or list of str
        Path(s) to training CSV file(s). If a list, files are concatenated.
    validSetPath : str or list of str
        Path(s) to validation CSV file(s).
    num_outputs : int
        Number of output columns in the CSV files.
    num_inputs : int or None
        Number of input columns. If None, auto-detected from metadata or CSV.

    Returns:
    --------
    data : tuple
        (XTrain, yTrain, XValid, yValid [, XTest, yTest])
    """
    # Auto-detect num_inputs if not specified
    if num_inputs is None:
        first_path = trainingSetPath if isinstance(trainingSetPath, str) else trainingSetPath[0]
        num_inputs = auto_detect_num_inputs(first_path, num_outputs)

    # Load training data (support single path or list of paths)
    if isinstance(trainingSetPath, list):
        XTrain, yTrain = _load_csv_multi(trainingSetPath, num_inputs, num_outputs, dtype)
    else:
        XTrain, yTrain = _load_csv_single(trainingSetPath, num_inputs, num_outputs, dtype)

    if isinstance(validSetPath, list):
        XValid, yValid = _load_csv_multi(validSetPath, num_inputs, num_outputs, dtype)
    else:
        XValid, yValid = _load_csv_single(validSetPath, num_inputs, num_outputs, dtype)

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

    AUTOTUNE = tf.data.AUTOTUNE

    trainSetBatch = (trainSet
                     .shuffle(min(batchSizeTrain * 2, inputLength))
                     .batch(batchSizeTrain)
                     .cache()
                     .prefetch(AUTOTUNE))
    validSetBatch = validSet.batch(batchSizeValid).cache().prefetch(AUTOTUNE)
    sets = (trainSetBatch, validSetBatch)
    if testSetExist:
        testSetBatch = testSet.batch(batchSizeValid).cache().prefetch(AUTOTUNE)
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
