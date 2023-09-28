import os
import numpy as np
import pandas as pd
from numba import njit

### Jitted Discretization Functions ###


@njit
def sigmoid_norm(series, mean, std):
    """
    Normalizes a series using a sigmoid function.
    Both mean and std must be provided.
    """
    normed = 1 / (1 + np.exp(-(series - mean) / std))
    return normed


@njit
def create_sigmoid_points(mean, std, num_points):
    """Creates points sampled from a sigmoid function defined by the given mean and std."""
    data = np.linspace(-7 * std + mean, 7 * std + mean, num_points)
    sigmoid_values = 1 / (1 + np.exp(-(data - mean) / std))
    return sigmoid_values


@njit
def inverse_sigmoid_norm(series, mean, std):
    """Inverse of sigmoid_norm."""
    return mean - std * np.log(1 / series - 1)


@njit
def get_discretized_series(normalized_series, sigmoid_points):
    """Returns discretized indices for the given normalized series based on the closest sigmoid points."""
    return np.argmin(np.abs(normalized_series[:, None] - sigmoid_points), axis=1)


@njit
def discretize(series, resolution):
    """
    Discretizes the given series using sigmoid normalization and the given resolution.
    Returns the discretized indices.
    """
    mean, std = np.mean(series), np.std(series)
    sigmoid_points = create_sigmoid_points(mean, std, resolution)
    normalized_series = sigmoid_norm(series, mean, std)
    discretized_series = get_discretized_series(normalized_series, sigmoid_points)
    return discretized_series, sigmoid_points, mean, std


### Symbolic Discretization Functions ###

COMMON_RESOLUTIONS = [256, 512, 1024, 2048]


def save_char_map_to_file(char_map, resolution):
    """Saves the character map to a .txt file for the given resolution."""
    path = os.path.join("./character_data", f"char_map_{resolution}.txt")
    with open(path, "w") as f:
        f.write(char_map)


def load_char_map_from_file(resolution):
    """Loads the character map from a .txt file for the given resolution."""
    path = os.path.join("./character_data", f"char_map_{resolution}.txt")
    with open(path, "r") as f:
        return f.read()


def precompute_common_char_maps():
    """Computes and saves char maps for common resolutions."""
    for resolution in COMMON_RESOLUTIONS:
        char_map = load_top_characters(
            "./character_data/fixed_unicode_freq.csv", resolution
        )
        save_char_map_to_file(char_map, resolution)


def load_top_characters(filename, num_chars):
    """Returns a string of the top characters from the given file."""
    df = pd.read_csv(filename)
    sorted_chars = df["char"].str.strip("'").tolist()

    # Filter out unwanted characters
    excluded_chars = [" ", '"', ",", ".", ";"]
    sorted_chars = [
        char for char in sorted_chars if len(char) == 1 and char not in excluded_chars
    ]

    # Cut off the list to the number of desired characters
    sorted_chars = sorted_chars[:num_chars]

    # Create the result list
    result = [""] * num_chars
    middle = num_chars // 2

    for i, char in enumerate(sorted_chars):
        if i % 2 == 0:
            # Even characters go to the right of middle
            result[middle + i // 2] = char
        else:
            # Odd characters go to the left of middle
            result[middle - (i // 2 + 1)] = char

    return "".join(result)


def get_char_map(resolution):
    """Returns a character map for the given resolution."""

    # Check if char map for this resolution already exists, if yes load from file
    path = os.path.join("./character_data", f"char_map_{resolution}.txt")
    if resolution in COMMON_RESOLUTIONS and os.path.exists(path):
        return load_char_map_from_file(resolution)

    # Otherwise, generate it on the fly
    return load_top_characters("./character_data/fixed_unicode_freq.csv", resolution)


### Main Discretization Functions ###


def discretize_encode(series, resolution, dtype="symbolic"):
    """
    Given a series, resolution, and a data type (either 'numerical' or 'symbolic'),
    returns relevant information based on the choice.
    """
    discretized_series, sigmoid_points, mean, std = discretize(series, resolution)

    # If the dtype is numerical, return the results right away
    if dtype == "numerical":
        return discretized_series, sigmoid_points, mean, std

    # Otherwise, perform symbolic operations
    char_map = get_char_map(resolution)
    discretized_characters = [char_map[index] for index in discretized_series]
    char_to_point_mapping = {char_map[i]: sigmoid_points[i] for i in range(resolution)}
    discretized_string = "".join(discretized_characters)

    return (
        discretized_string,
        char_to_point_mapping,
        discretized_series,
        sigmoid_points,
        mean,
        std,
    )


def discretize_decode(series, mean, std, char_to_point_mapping=None, dtype="symbolic"):
    """
    Decodes an encoded time series, whether symbolic or numerical.
    """

    if dtype == "numerical":
        return inverse_sigmoid_norm(series, mean, std)

    normalized_values = np.array([char_to_point_mapping[char] for char in series])
    # Invert the sigmoid normalization
    inverted_values = inverse_sigmoid_norm(normalized_values, mean, std)
    return inverted_values
