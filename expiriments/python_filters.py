from lane_detection.utils.expirements.box_blur import BoxBlur
from lane_detection.utils.expirements.gaussian import GaussianBlur
from lane_detection.utils.expirements.median_blur import MedianBlur
from lane_detection.utils.expirements.bilateral_blur import BilateralBlur

class PythonFilters:

    def __init__(self, ksize, sigma_color, sigma_space, sigma_x:float = 0.0, sigma_y:float = 0.0):
        
        self.box = BoxBlur(ksize)
        self.gaussian = GaussianBlur(ksize, sigma_x, sigma_y)
        self.median = MedianBlur(ksize)
        self.bilateral = BilateralBlur(ksize, sigma_color, sigma_space)

    def blur(self, frame, blur_type):
        if blur_type == "gaussian":
            frame = self.gaussian.blur(frame)
        elif blur_type == "box":
            frame = self.box.blur(frame)
        elif blur_type == "median":
            frame = self.median.blur(frame)
        elif blur_type == "bilateral":
            frame = self.bilateral.blur(frame)
        return frame

if __name__=="__main__":
    import cv2
    import numpy as np

    source = "../../img/mri-skull-40-percent-gaussian-noise.jpg"
    src = cv2.imread(source)
    if src is None:
        raise ValueError(f"ERROR: Could not read source {source}")

    h, w = src.shape[:2]
    ratio = 500 / w
    w = 500
    h = int(ratio * h)
    src = cv2.resize(src, (w, h), interpolation=cv2.INTER_AREA)

    filter = PythonFilters(5, 50, 100, 0., 0.)

    box = filter.blur(src, blur_type="box")
    gaussian = filter.blur(src, blur_type="gaussian")
    median = filter.blur(src, blur_type="median")
    bilateral = filter.blur(src, blur_type="bilateral")

    top = np.hstack([src, src, src, src])
    bottom = np.hstack([box, median, gaussian, bilateral])

    final = np.vstack([top, bottom])

    cv2.imshow("Side-by-Side", final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()