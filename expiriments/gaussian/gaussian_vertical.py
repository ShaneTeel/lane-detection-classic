import numpy as np
import math

class GaussianVertical:
    def __init__(self, radius, sigma):
        self.radius = radius
        self.sigma = self._get_sigma() if sigma == 0 else sigma
        self.weights = self._get_weights_1d()
        self.weight_sum = sum(self.weights)

    def blur(self, frame):
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)

        for col in range(w):
            for row in range(self.radius, h - self.radius):

                weighted_sum = 0

                for offset in range(-self.radius, self.radius + 1):
                    weight = self.weights[offset + self.radius]
                    pixel = frame[row + offset, col]
                    weighted_sum += weight * pixel
                result[row, col] = weighted_sum / self.weight_sum
        return result.astype(np.uint8)

    def _get_weights_1d(self):
        weights = []

        for offset in range(-self.radius, self.radius + 1):
            val = math.exp(-offset**2 / (2 * self.sigma**2))
            weights.append(val)

        return weights
    
    def _get_sigma(self):
        return 0.3 * ((self.radius * 2) * 0.5 - 1) + 0.8