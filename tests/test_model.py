import numpy as np
from treenet import TreeNet

def test_model_training_and_prediction():
    X_train = np.random.rand(10, 5)
    y_train = np.random.randint(0, 3, size=(10,))
    X_test = np.random.rand(5, 5)

    model = TreeNet(layer_count=3, breath_count=2)
    model.train(X_train, y_train)

    probs = model.predict_prob(X_test)
    assert probs.shape == (5, 3)

    preds = model.predict(X_test)
    assert preds.shape == (5,)
