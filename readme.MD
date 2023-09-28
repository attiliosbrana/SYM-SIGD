# SYM-SIGD: SYMbolic SIGmoid Discretizer

## Overview

SYM-SIGD offers a novel approach to data discretization by utilizing sigmoid standardization, ensuring zero information loss. After the discretization using the sigmoid function, a symbolic representation of the data is obtained, which can be leveraged in various ways, such as feeding information to large language models. Additionally, this project provides decoding functions to easily undo the discretizations.

### Example Demonstration of SYM-SIGD Discretization ###

Given an original time series:

```python
original_series = np.array([47.74208669, 47.80864911, 47.87534023, 47.94216033, 48.00910966,
                           48.0761885 , 48.14339711, 48.21073577, 48.27820473, 48.34580428])
```

After applying SYM-SIGD, we get a symbolic representation of the series:

```python
symbolic_discretization = 'Гრრრहह日日日ص'
```

Decoding the symbolic representation, we obtain:

```python
decoded_discretization = np.array([47.72336171, 47.88323436, 47.88323436, 47.88323436, 48.04310701,
                                  48.04310701, 48.20297966, 48.20297966, 48.20297966, 48.36285231])
```

This showcases the core functionality of the SYM-SIGD process. Though there's a slight variation between the original and decoded series, the overall shape and trend of the data is retained, making it useful for various applications where precision is not the absolute priority, but where data compression and symbolic representation are of value.

## Features

- **Sigmoid Normalization**: Standardizes data without information loss.
- **Sigmoid-Based Discretization**: Transforms data based on sigmoid points.
- **Symbolic Representation**: Converts discretized data into a symbolic format using common Unicode characters.
- **Decoding Functions**: Allows for effortless reconstruction of original data.

## Dataset

- `fixed_unicode_freq.csv`: This file lists the most common Unicode characters based on their Wikipedia frequency. The data source is credited to [Andrea Esuli's Wikipedia tools](http://www.esuli.it/).

## Usage

### 1. Discretization

Use the `discretize_encode` function from `discretize.py`:
```python
from discretize import discretize_encode

# Parameters: series (numpy array), resolution (int), dtype ("symbolic" or "numerical")
output = discretize_encode(series, resolution, dtype="symbolic")
```

### 2. Decoding

To recover the original data from its symbolic or numerical representation, use the `discretize_decode` function:
```python
from discretize import discretize_decode

# Parameters: series (string or numpy array), mean (float), std (float), char_to_point_mapping (dict, optional), dtype ("symbolic" or "numerical")
recovered_data = discretize_decode(series, mean, std, char_to_point_mapping=None, dtype="symbolic")
```

## Example

An illustrative example is provided in `example.ipynb` which showcases the entire process from discretizing a given time series to recovering it, and then comparing the original versus the recovered series.

## Dependencies

- numpy
- pandas
- numba
- seaborn
- matplotlib

## License

MIT License