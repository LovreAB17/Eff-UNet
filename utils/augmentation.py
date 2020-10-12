import albumentations as albu
from utils.utils import to_tensor


def get_training_augmentation():
    """
    Methods for training augmentation
    """

    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.GridDistortion(p=0.5),
        albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """
    Methods for validation augmentation
    """

    transforms = [albu.HorizontalFlip(p=0.0)]
    return albu.Compose(transforms)

