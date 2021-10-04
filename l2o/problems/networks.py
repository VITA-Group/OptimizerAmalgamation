"""NN based training problems."""

import functools

import tensorflow as tf
import tensorflow_datasets as tfds

from .problem import Problem
from .stateless_keras import (
    Dense, Sequential, Conv2D, MaxPooling2D, AveragePoolingAll)


def load_images(dataset, split="train"):
    """Load images and cast to float between 0 and 1.

    Note: shuffle_files MUST be false, since shuffling with seeds occurs later
    in the pipeline.

    Suggested Datasets:
    +---------------+-----------+-----+---------------------------------------+
    | dataset       | shape     | k   | description                           |
    +---------------+-----------+-----+---------------------------------------+
    | cifar10       | 32x32x3   | 10  | images                                |
    | emnist        | 28x28     | 10  | handwritten digits                    |
    | fashion_mnist | 28x28     | 10  | images (clothing)                     |
    | kmnist        | 28x28     | 10  | handwritten Japanese characters       |
    | mnist         | 28x28     | 10  | handwritten digits                    |
    | cifar100      | 32x32x3   | 100 | images                                |
    | omniglot      | 105x105x3 | 50  | handwritten characters                |
    | stl10         | 96x96x3   | 10  | images                                |
    +---------------+-----------+-----+---------------------------------------+

    Parameters
    ----------
    dataset : str
        Dataset name in tfds.
    split : str
        "train" or "test".

    Returns
    -------
    [tf.data.Dataset, tfds.core.DatasetInfo]
        [0] Loaded dataset
        [1] tfds info object.
    """
    dataset, info = tfds.load(
        dataset, split=split, shuffle_files=True,
        with_info=True, as_supervised=True)

    def _cast(x, y):
        return tf.cast(x, tf.float32) / 255., y

    return dataset.map(_cast), info


def _make_tfds(network, dataset="mnist", **kwargs):
    """Create training problem using tensorflow_datasets."""
    dataset, info = load_images(dataset)

    try:
        input_shape = info.features['image'].shape
        labels = info.features['label'].num_classes
    except KeyError:
        raise TypeError("Dataset must have 'image' and 'label' features.")
    except AttributeError:
        raise TypeError(
            "'image' feature must have a shape, and 'label' feature must have"
            + "a number of classes num_classes.")
    if None in input_shape:
        raise TypeError("Dataset does not have fixed input dimension.")

    return Problem(
        network(input_shape, labels), dataset,
        tf.keras.losses.SparseCategoricalCrossentropy(
            reduction=tf.keras.losses.Reduction.NONE),
        size=info.splits['train'].num_examples, **kwargs)


def mlp_classifier(
        dataset="mnist", layers=[128, ], activation="relu", **kwargs):
    """Create MLP classifier training problem.

    Keyword Args
    ------------
    dataset : str
        Dataset from tdfs catalog. Must have fixed input dimension and
        output labels.
    layers : int[]
        Array of hidden layer sizes for MLP.
    activation : str
        Keras activation type
    **kwargs : dict
        Passed on to Classifier()

    Returns
    -------
    problem.Problem
        Created problem

    Raises
    ------
    KeyError
        Selected dataset does not have an image or label.
    AttributeError
        Image does not specify shape or label does not specify num_classes.
    TypeError
        Dataset does not have a fixed input dimension.
    """
    if isinstance(activation, str):
        activation = tf.keras.activations.get(activation)

    def _network(input_shape, labels):
        return Sequential(
            [Dense(u, activation=activation) for u in layers]
            + [Dense(labels, activation=tf.nn.softmax)], input_shape)

    return _make_tfds(_network, dataset=dataset, **kwargs)


def conv_classifier(
        dataset="mnist", layers=[(5, 32, 2), ], head_type="dense",
        activation=tf.nn.relu, **kwargs):
    """Create Convolutional classifier training problem.

    Keyword Args
    ------------
    dataset : str
        Dataset from tdfs catalog. Must have fixed input dimension and
        output labels.
    layers : int[][3]
        List of (num_filters, kernel_size, stride) for convolutional layers
    head_type : str
        Classification head type. Can be "dense" (flatten + sigmoid dense) or
        "average" (average pooling over all dimensions). For "average", the
        number of filters in the last layer is overwritten with the number
        of classes.
    activation : str or dict
        Keras activation type. If dict, uses tf.keras.get-like syntax
    **kwargs : dict
        Passed on to Classifier()

    Returns
    -------
    problem.Problem
        Created problem

    Raises
    ------
    KeyError
        Selected dataset does not have an image or label.
    AttributeError
        Image does not specify shape or label does not specify num_classes.
    TypeError
        Dataset does not have a fixed input dimension.
    """
    # NOTE: tf.keras.activations.get is broken as of tf 2.3 for dict
    # deserializing (with config). As such, we manually reimplement its
    # supposed functionality here.
    if isinstance(activation, str):
        activation = tf.keras.activations.get(activation)
    elif isinstance(activation, dict):
        activation = functools.partial(
            tf.keras.activations.get(activation["class_name"]),
            **activation["config"])

    def _preprocess(img):
        shape = img.shape.as_list()
        return tf.cast(tf.reshape(img, shape[:-1] + []), tf.float32) / 255.

    def _deserialize(args):
        if isinstance(args, int):
            return MaxPooling2D(pool_size=(args, args))
        elif isinstance(args, (list, tuple)) and len(args) == 3:
            f, k, s = args
            return Conv2D(f, k, stride=s, activation=activation)
        else:
            raise TypeError("Not a valid layer: {}".format(args))

    if head_type == "dense":
        def _network(input_shape, labels):
            return Sequential(
                [_deserialize(x) for x in layers]
                + [Dense(labels, activation=tf.nn.softmax)], input_shape)
    elif head_type == "average":
        def _network(input_shape, labels):
            return Sequential(
                [_deserialize(x) for x in layers[:-1]]
                + [Conv2D(
                    labels, layers[-1][1], stride=layers[-1][2],
                    activation=tf.nn.softmax)]
                + [AveragePoolingAll()], input_shape)
    else:
        raise ValueError(
            "Invalid classification head type {}. "
            "Must be 'dense' or 'average'.".format(head_type))

    return _make_tfds(_network, dataset=dataset, **kwargs)
