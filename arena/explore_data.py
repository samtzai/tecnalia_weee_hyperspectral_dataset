
import numpy
import scipy.io
import skimage, skimage.io
import glob
import os
import matplotlib.pyplot as plt

# list_files =glob.glob('dataset/*.mat')

# print(list_files)
# # list_files = [
# # "dataset/Fast10_Steel_Al_061018_13ms_288_1120_00",
# # "dataset/Cu_Brass_Al_10ms_360_1704_01",
# # "dataset/White_Cu_Al_Brass_25ms_19fps_OL_378_610_00"
# # ]

list_files = [
    'dataset/data/Fast10_Steel_Al_061018_13ms_288_1120_00.mat',  # parece mal calibrado
    'dataset/data/Fast10_Steel_Al_061018_13ms_284_1128_02.mat', 
    'dataset/data/Fast10_Steel_Al_061018_13ms_300_1132_01.mat',
    'dataset/data/Fast10_Steel_Al_061024_13ms_refl_228_1240_01.mat',
    'dataset/data/Fast10_Steel_Al_061024_13ms_refl_256_1252_02.mat',
    'dataset/data/Fast10_Steel_Al_061024_13ms_refl_292_1228_00.mat',
    'dataset/data/Cu_Brass_Al_10ms_360_1704_01.mat', 
    'dataset/data/White_Cu_Al_Brass_25ms_19fps_OL_318_616_01.mat', 
    'dataset/data/061025_Steel_Cu_Al_Steel_Brass_25ms_20fps_288_720_01.mat', 
    'dataset/data/White_Cu_Al_Brass_25ms_19fps_OL_378_610_00.mat', 
    'dataset/data/Cu_Brass_Al_10ms_412_1704_00.mat',
    'dataset/data/Cu_Brass_Al_MIXED_10ms_normalized_452_1404_01.mat',
    'dataset/data/Cu_Brass_Al_MIXED_10ms_normalized_492_1392_00.mat'
    ]

positions = [
    {'j': 788, 'i': 273}, #AL
    {'j': 902, 'i': 148}, #AL
    {'j': 814, 'i': 224 }, #AL
    {'j': 886, 'i': 136}, #AL
    {'j': 1043, 'i': 118}, #AL
    {'j': 907, 'i': 214 }, #AL
    {'j': 924, 'i': 200}, #BR
    {'j': 488, 'i': 172}, #BR
    {'j': 665, 'i': 230}, #BR
    {'j': 501, 'i': 211}, #BR
    {'j': 799, 'i': 391}, #BR
    {'j': 863, 'i': 255}, #BR
    {'j': 400, 'i': 400}, #BR
]

plt.figure()
wl = numpy.linspace(415.05, 1008.10, 76)
for file,points in zip(list_files,positions):
    filename = file.rsplit('.', 1)[0]
    #Load mat file
    data = scipy.io.loadmat(file)['hyperfile']
    #Load Ground Truth
    # data_gt = skimage.io.imread('%s_GT.bmp' % filename)

    # plt.imshow(data_gt)
    # plt.show(block = True)
    # get point from mouse click
    max_val = data[:].max()
    if data.shape[2] >= 76:
        data = data[:,:,-76:]
    if data[:].max()>500:
        data = data / 4096
    # Print file name and number of channels
    spectra = data[points['i'], points['j'], :]
    
    scipy.io.savemat('%s_corrected.mat' % filename,{'hyperfile':data, 'wavelengths':wl})
    plt.plot(spectra)
    print(".................................")
    print(file,max_val, spectra.shape,'index 20:',spectra[20], 'index 40:',spectra[40], 'index 60:',spectra[60] )
    

plt.show(block=True)
plt.savefig('test.png')



# dataset_name = 'bandselection'
# dataset_name2 = 'White_Cu_Al_Brass_25ms_19fps_OL_378_610_00'
# # load dataset mat file
# data = scipy.io.loadmat('dataset/%s.mat' % dataset_name)
# data2 = scipy.io.loadmat('dataset/%s.mat' % dataset_name2)['Hyperfile']
# print(data)
# # load labels mat file
# data_gt = skimage.io.imread('dataset/%s_gt.bmp' % dataset_name)
# print(data_gt)
# pass


# skimage.io.imsave('dataset/%s.png' % dataset_name, data[:,:,10])
# # for each channel in the gt_ save binary imag emask
# for i in range(data_gt[:].max() + 1):
#     # save the binary image
#     skimage.io.imsave('dataset/%s_gt_%d.png' % (dataset_name, i), data_gt[:, :] == i)
# # save a pseudocolor figure with the ground truth
# import matplotlib.pyplot as plt
# plt.imshow(data_gt)
# plt.show()
# plt.savefig('dataset/%s_gt.png' % dataset_name)



# save an image with the ground truth

print('stop')
