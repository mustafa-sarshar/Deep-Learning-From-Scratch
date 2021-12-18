# In[] Libs
import numpy as np
from neural_network_classes.layers import Dense

# In[] Run the App
if __name__ == "__main__":
    X = np.array([
        np.random.random(5),
        np.random.random(5),
        np.random.random(5),
        np.random.random(5),
        np.random.random(5)
    ])
    print(f"X:\n{X}")
    layer_dense_1 = Dense(n_inputs=5, n_neurons=10)
    layer_dense_2 = Dense(n_inputs=10, n_neurons=4)
    print(layer_dense_1.weights.shape)
    print(layer_dense_2.weights.shape)

    layer_dense_1.forward(x=X)
    print(f"y_1:\n{layer_dense_1.y}")

    layer_dense_2.forward(x=layer_dense_1.y)
    print(f"y_2:\n{layer_dense_2.y}")
