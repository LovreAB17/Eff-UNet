import cv2

from torch.utils.data import Dataset
from utils.utils import to_tensor


class CustomDataset(Dataset):
    def __init__(self, images_list, masks_list, transforms):
        self.image_files = images_list
        self.masks_files = masks_list
        self.transforms = transforms

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        mask_path = self.masks_files[idx]
        img = cv2.imread(image_path)
        mask = cv2.imread(mask_path)

        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        return to_tensor(img), to_tensor(mask)

    def __len__(self):
        return len(self.image_files)
