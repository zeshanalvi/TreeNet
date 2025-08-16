# TreeNet

TreeNet is a lightweight and customizable machine learning model designed for flexible experimentation with layered network structures. It provides simple training and prediction interfaces and is well-suited for rapid prototyping and educational purposes.

## Installation
You can install the package directly from PyPI:

```bash
pip install treenet
```

Or install it from the source:

```bash
git clone https://github.com/zeshanalvi/treenet.git
cd treenet
pip install .
```


## Usage
Here’s a quick example of how to use ```TreeNet```:

```python
import numpy as np
from treenet import TreeNet

# Initialize the model
model = TreeNet(layer_count=3, breath_count=2)

# Sample training data
trainX = np.random.rand(100, 8)   # 100 samples, 8 features
trainY = np.random.randint(0, 2, size=(100, 1))  # Binary labels

# Train the model
model.train(trainX, trainY)

# Sample test data
testX = np.random.rand(10, 8)

# Predict probabilities
probabilities = model.predict_prob(testX)
print("Probabilities:\n", probabilities)

# Predict labels
predictions = model.predict(testX)
print("Predictions:\n", predictions)

```
## API Reference
```TreeNet```
```python
__init__(self, layer_count=2, breath_count=1)
```

Initializes a new TreeNet model.

```layer_count (int):``` Number of layers in the network (default: 2).

```breath_count (int):``` Number of parallel branches per layer (default: 1).

```train(self, trainX, trainY)```

Trains the model on given data.

```trainX (np.ndarray):``` Training features of shape ```(n_samples, n_features)```.

```trainY (np.ndarray):``` Training labels of shape ```(n_samples, 1)```.

```predict_prob(self, testX) -> np.ndarray```

Returns the probability distribution over classes for input samples.

```testX (np.ndarray):``` Test features of shape ```(n_samples, n_features)```.

Returns: ```np.ndarray``` of shape ```(n_samples, n_classes)```.

```predict(self, testX) -> np.ndarray```

Returns predicted class labels for input samples.

```testX (np.ndarray):``` Test features of shape ```(n_samples, n_features)```.

Returns: ```np.ndarray``` of shape ```(n_samples, 1)```.

## Project Structure

```
treenet/
│
├── treenet/
│   ├── __init__.py
│   ├── model.py         # Implementation of TreeNet class
│
├── tests/
│   ├── test_model.py    # Unit tests for TreeNet
│
├── README.md
├── setup.py
├── pyproject.toml
├── LICENSE
```