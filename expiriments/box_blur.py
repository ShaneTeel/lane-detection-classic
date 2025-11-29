import numpy as np

class BoxBlur:

    def __init__(self, ksize):

        self.ksize = ksize
        self.radius = ksize // 2
    
    def blur(self, src):
        
        if len(src.shape) == 2:
            channels = [src]
        else:
            channels = [src[:, :, i] for i in range(src.shape[2])]

        results = []

        for frame in channels:
            result = self._horizontal_pass(frame)
            result = self._vertical_pass(result)
            results.append(result)

        if len(results) == 1:
            return results[0].astype(np.uint8)
        else:
            return np.dstack(results).astype(np.uint8)
        
    def _horizontal_pass(self, frame):
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        for row in range(self.radius, h - self.radius):
            running_sum = 0.0
            for col in range(self.ksize):
                running_sum += frame[row, col]

            result[row, self.radius] = running_sum / self.ksize

            for col in range(self.radius + 1, w - self.radius):
                running_sum -= frame[row, col - self.radius - 1]
                running_sum += frame[row, col + self.radius]
                result[row, col] = running_sum / self.ksize

        return result 

    def _vertical_pass(self, frame):
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        for col in range(self.radius, w - self.radius):
            running_sum = 0.0
            for row in range(self.ksize):
                running_sum += frame[row, col]
            result[self.radius, col] = running_sum / self.ksize

            for row in range(self.radius + 1, h - self.radius):
                running_sum -= frame[row - self.radius -1, col]
                running_sum += frame[row + self.radius, col] 
                result[row, col] = running_sum / self.ksize
        return result