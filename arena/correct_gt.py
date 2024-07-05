import numpy as np
import skimage, skimage.io
import os
import matplotlib.pyplot as plt

path = 'dataset/data'

list_files = os.listdir(path)

num_classes = 7

for file in list_files:
    # if not 'test' in file:
    #     continue
    if not '_gt' in file :
        continue
  
    image = skimage.io.imread(os.path.join(path, file))
    max_val = image[:].max()
    print(file,max_val)
    image[image>=num_classes] = 0
    print(image[:].max())
    if max_val >= num_classes:
        skimage.io.imsave(os.path.join(path, file), image)
    

# move GT 32 pixels up
# move_up = 32
# for file in list_files:
#     # if not 'test' in file:
#     #     continue
#     if not '_gt' in file :
#         continue
#     if not 'Cu_Brass_Al_10ms' in file:
#         continue
  
#     image_gt = skimage.io.imread(os.path.join(path, file))
#     corrected_gt = np.zeros_like(image_gt)

#     corrected_gt[:-move_up,:,:] = image_gt[move_up:,:,:]

#     skimage.io.imsave(os.path.join(path, file), corrected_gt)
    
