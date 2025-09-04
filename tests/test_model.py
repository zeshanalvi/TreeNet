import numpy as np, pandas as pd, string, random

from treenet import TreeNet

def test_model_training_and_prediction():
    y=4
    n_samples=100
    col_labels = [''.join(random.choices(string.ascii_uppercase, k=3)) for _ in range(y)]

    X_train = pd.DataFrame(np.random.rand(n_samples, y), columns=col_labels)
    X_test = pd.DataFrame(np.random.rand(n_samples, y), columns=col_labels)
    y_train = np.random.randint(0, y, size=(n_samples,))


    model = TreeNet(layer_count=3, breath_count=2)
    model.train(X_train, y_train)

    probs = model.predict_prob(X_test)
    assert probs.shape == (n_samples, y)

    preds = model.predict(X_test)
    assert preds.shape == (n_samples,1)

def test_model_dataframes():
    y=4
    n_samples=100
    col_labels = [''.join(random.choices(string.ascii_uppercase, k=3)) for _ in range(y)]

    X_train = pd.DataFrame(np.random.rand(n_samples, y), columns=col_labels)
    X_test = pd.DataFrame(np.random.rand(n_samples, y), columns=col_labels)
    y_train = pd.DataFrame(np.random.randint(0, y, size=(n_samples,)))


    model = TreeNet(layer_count=3, breath_count=2)
    model.train(X_train, y_train)

    probs = model.predict_prob(X_test)
    assert probs.shape == (n_samples, y)

    preds = model.predict(X_test)
    assert preds.shape == (n_samples,1)

def test_model_ndarray():
    y=4
    n_samples=100
    X_train = np.random.rand(n_samples, y)
    X_test = np.random.rand(n_samples, y)
    y_train = np.random.randint(0, y, size=(n_samples,))


    model = TreeNet(layer_count=3, breath_count=2)
    model.train(X_train, y_train)

    probs = model.predict_prob(X_test)
    assert probs.shape == (n_samples, y)

    preds = model.predict(X_test)
    assert preds.shape == (n_samples,1)