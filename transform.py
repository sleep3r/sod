import albumentations as albu
from albumentations.pytorch.transforms import ToTensorV2


def get_test_augmentation(img_size):
    transforms = albu.Compose([
        albu.Resize(img_size, img_size, always_apply=True),
        albu.Normalize([0.485, 0.456, 0.406],
                       [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return transforms