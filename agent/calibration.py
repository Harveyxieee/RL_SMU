## first edition: global conformal
import torch

class ResidualCalibrator:
    def __init__(
        self,
        alpha=0.1,
        min_epsilon=1e-4,
        max_epsilon=0.5,
        scale=1.0,
        mode="global_quantile",
    ):
        self.alpha = alpha
        self.min_epsilon = min_epsilon
        self.max_epsilon = max_epsilon
        self.scale = scale
        self.mode = mode
        self.residuals = None

    def fit(self, residuals):
        self.residuals = residuals.detach().cpu()
        q = torch.quantile(self.residuals, 1 - self.alpha)
        self.global_epsilon = float(q * self.scale)
        self.global_epsilon = max(self.min_epsilon, min(self.global_epsilon, self.max_epsilon))

    def predict_epsilon(self, states, actions):
        batch_size = states.shape[0]
        return torch.full(
            (batch_size,),
            self.global_epsilon,
            device=states.device,
            dtype=states.dtype,
        )