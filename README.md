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
import numpy as np, pandas as pd, random, string
from treenet import TreeNet

# Initialize the model
model = TreeNet(layer_count=3, breath_count=2)

# Sample training data
col_labels = [''.join(random.choices(string.ascii_uppercase, k=10)) for _ in range(20)] # labels of 20 columns
trainXnp = np.random.rand(100, 20)   # 100 samples, 20 features
trainX = pd.DataFrame(trainXnp, columns=col_labels) # Dataframe
trainY = np.random.randint(0, 8, size=(100,)) # Multiclass labels with 8 classes

# Train the model
model.train(trainX, trainY)

# Sample test data
testXnp = np.random.rand(10, 20) # 10 samples, 20 features
testX = pd.DataFrame(testXnp, columns=col_labels) # Dataframe

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

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

TreeNet was built to simplify experimentation with layered machine learning structures and provide a lightweight package for educational and prototyping use.