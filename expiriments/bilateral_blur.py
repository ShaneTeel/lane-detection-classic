import numpy as np
import math

class BilateralBlur:

    def __init__(self, diam, sigma_color, sigma_space):
        self.diam = diam
        self.radius = diam // 2
        self.sigma_color = sigma_color
        self.sigma_space = sigma_space
        self.spatial_weights = self._get_spatial_weights()
    
    def blur(self, frame):
        
        h, w = frame.shape[:2]

        if len(frame.shape) == 2:
            channels = [frame]
        else:
            channels = [frame[:, :, i] for i in range(frame.shape[2])]
        
        results = []
        for frame in channels:
            result = np.zeros_like(frame)
            for row in range(self.radius, h - self.radius):
                for col in range(self.radius, w - self.radius):
                    pixel = float(frame[row, col])
                    
                    weighted_sum = 0.0

                    weight_total = 0.0

                    for kernel_row, neighbor_row in enumerate(range(row - self.radius, row + self.radius + 1)):
                        for kernel_col, neighbor_col in enumerate(range(col - self.radius, col + self.radius + 1)):
                            neighbor = float(frame[neighbor_row, neighbor_col])

                            spatial_weight = self.spatial_weights[kernel_row, kernel_col]

                            diff = (neighbor - pixel)**2

                            color_weight = math.exp(
                                -diff / (2 * self.sigma_color**2)
                            )
                            weight = spatial_weight * color_weight
                            weighted_sum += neighbor * weight

                            weight_total += weight
                            
                    result[row, col] = weighted_sum / weight_total

            results.append(result)
        if len(results) == 1:
            return results[0].astype(np.uint8)
        else:
            return np.dstack(results).astype(np.uint8)
    
    def _get_spatial_weights(self):
        weights = np.zeros((self.diam, self.diam))
        for row in range(self.diam):
            for col in range(self.diam):
                dist_row = row - self.radius
                dist_col = col - self.radius
                weights[row, col] = math.exp(
                    -(dist_row**2 + dist_col**2) / (2 * self.sigma_space**2)
                )
        return weights