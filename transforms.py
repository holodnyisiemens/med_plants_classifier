import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2
import cv2
import numpy as np

# pytorch lightning works only with named arguments and numpy array
class Transform:
    def __init__(self, transform: A.Compose):
        self.transform = transform

    def __call__(self, image, *args, **kwargs):
        return self.transform(image=np.array(image))

class RotateToVertical(ImageOnlyTransform):
    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        if img.shape[0] < img.shape[1]:
            return cv2.transpose(img)
        return img

# values for A.Normalize from https://pytorch.org/hub/pytorch_vision_densenet/
train_transform = A.Compose(
    [
        RotateToVertical(p=1.0),
        A.Resize(height=320, width=180, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0),
        ToTensorV2(transpose_mask=False, p=1.0),
    ]
)

val_transform = A.Compose(
    [
        RotateToVertical(p=1.0),
        A.Resize(height=320, width=180, p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0,
                    p=1.0),
        ToTensorV2(transpose_mask=False, p=1.0),
    ]
)
