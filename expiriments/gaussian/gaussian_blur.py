import numpy as np
from .gaussian_horizontal import GaussianHorizontal
from .gaussian_vertical import GaussianVertical

class GaussianBlur:

    def __init__(self, ksize:int = 5, sigma_x:float = 0., sigma_y:float = 0.):
        if ksize % 2 == 0 or ksize < 3 or not isinstance(ksize, int):
            raise ValueError(f"ERROR: Argument passed for 'ksize' param ({ksize}) must be an odd float greater than 1.")

        self.ksize = ksize
        self.radius = ksize // 2
        self.horizontal = GaussianHorizontal(self.radius, sigma_x)
        self.vertical = GaussianVertical(self.radius, sigma_y)

    def blur(self, frame):

        if len(frame.shape) == 2:
            channels = [frame]
        else:
            channels = [frame[:, :, i] for i in range(frame.shape[2])]
        
        results = []
        for frame in channels:
            result = self.horizontal.blur(frame)
            result = self.vertical.blur(result)
            results.append(result)
        
        if len(results) == 1:
            return results[0].astype(np.uint8)
        else:
            return np.dstack(results).astype(np.uint8)