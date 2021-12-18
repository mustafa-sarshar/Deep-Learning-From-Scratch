import numpy as np

class Loss:
    def calculate(self, y_pred, y_true):
        sample_losses = self.forward(y_pred, y_true)
        return np.mean(sample_losses) 

class CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        n_samples = len(y_pred)
        y_pred_clipped = np.clip(a=y_pred, a_min=1e-7, a_max=1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(n_samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis=-1)
        
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

class Accuracy:
    def calculate(self, y_pred, y_true):
        preds = np.argmax(y_pred, axis=-1)
        return np.mean(preds == y_true)


# In[] Test the Code
if __name__ == "__main__":
    y_true = np.array([
        [.1, .2, .3, .4, .5],
        #[.1, .2, .3, .4, .5]
    ])
    y_pred = np.array([
        [.1, .3, .3, .7, .0],
        #[.1, .1, .1, .7, .0]
    ])
    err = CategoricalCrossentropy()
    print("Loss:", err.calculate(y_pred, y_true))
