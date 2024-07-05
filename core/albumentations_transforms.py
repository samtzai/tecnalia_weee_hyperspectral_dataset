import set_python_path
import random
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_transforms(is_train,random_crop, input_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],max_pixel_value=1.0):
    transforms = []
    if is_train:
        transforms.append(A.Flip(p=0.5))
        # transforms.append(A.RandomRotate90(p=0.3))
        transforms.append(A.Transpose(p=0.5))
        # transforms.append(A.RandomScale(scale_limit=(-0.3,0), p=0.5, interpolation=1))
        # transforms.append(A.PadIfNeeded(min_height=input_size, min_width=input_size, border_mode=cv2.BORDER_REPLICATE))
    if random_crop:
        transforms.append(A.RandomCrop(input_size, input_size, p=1.0))
    
    # transforms.append(A.Resize(height=input_size, width=input_size, p=1.0))
    transforms.append(A.Normalize(mean=mean, std=std, max_pixel_value=max_pixel_value, p=1.0))    
    transforms.append(ToTensorV2(transpose_mask = True))
    
    return A.Compose(transforms)

