import numpy as np

class MedianBlur:

    def __init__(self, ksize):
        self.ksize = ksize
        self.radius = ksize//2
        self.kernel_area = ksize * ksize

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
                hist = np.zeros(256, dtype=int)

                for kernel_row in range(row - self.radius, row + self.radius + 1):
                    for kernel_col in range(0, self.ksize):
                        baseline_intensity = frame[kernel_row, kernel_col]
                        hist[baseline_intensity] += 1

                result[row, self.radius] = self._get_median(hist)

                for kernel_col in range(self.radius + 1, w - self.radius):
                    leaving = kernel_col - self.radius - 1
                    entering = kernel_col + self.radius
                    for kernel_row in range(row - self.radius, row + self.radius + 1):
                            intensity_to_remove = frame[kernel_row, leaving]
                            hist[intensity_to_remove] -= 1

                            new_intensity = frame[kernel_row, entering]
                            hist[new_intensity] += 1

                    result[row, kernel_col] = self._get_median(hist)
            results.append(result)

        if len(results) == 1:
            return results[0].astype(np.uint8)
        else:
            return np.dstack(results).astype(np.uint8)
    
    def _get_median(self, hist):
        count = 0
        half = self.kernel_area // 2
        for intensity in range(len(hist)):
            count += hist[intensity]
            if count > half:
                return intensity
        return 0