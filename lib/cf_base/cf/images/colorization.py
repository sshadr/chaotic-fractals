import numpy as np
import cv2

#==============================================

# apply a custom red-blue color map to a signed image
def apply_signed_colormap(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.tile(img[..., None], (1, 1, 3))
    return np.where(img > 0, img * np.array([[[1, 0, 0]]]), -img * np.array([[[0, 0, 1]]]))


#==============================================
