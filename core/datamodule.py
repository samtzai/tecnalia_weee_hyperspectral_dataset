from enum import Enum
import set_python_path
import os
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset,DataLoader
from core.utils.io import load_params, read_json
import skimage, skimage.io
import torch
import numpy
import os
import scipy

class BalancingMethods(int, Enum):
    no_balancing = 1
    ignore_last_class = 2
    substract_abundances = 3
    invert_abundances = 4
    effective_number_samples = 5 
    decrease_background = 6 

    @classmethod
    def from_string(cls, string):
        if string == 'no_balancing':
            return cls.no_balancing
        elif string == 'ignore_last_class':
            return cls.ignore_last_class
        elif string == 'substract_abundances':
            return cls.substract_abundances
        elif string == 'invert_abundances':
            return cls.invert_abundances
        elif string == 'effective_number_samples':
            return cls.effective_number_samples
        elif string == 'decrease_background':
            return cls.decrease_background
        else:
            raise ValueError('Invalid balancing method: {}'.format(string))
        
class SegmentationDataset(Dataset):
    def __init__(self,image_list, list_classes, class_indices, transforms=None, load_in_memory = False, wl_indices = None, num_image_draws_per_epoch = 1):
        # List of categories with their corresponding indexes
        self.classes=np.array(list_classes)
        self.class_indices=np.array(class_indices)
        self.image_list=np.array(image_list)
        self.wl_indices = wl_indices
        self.loaded_dataset_images = None
        self.transforms = transforms
        self.num_image_draws_per_epoch = num_image_draws_per_epoch
        
        if load_in_memory:
            self.loaded_dataset_images = []
            for image in self.image_list:
                self.loaded_dataset_images.append(self._load_file(image))

    def _load_file(self, file_path):
        file_id = file_path.split('.')[0]
        mat_file_path = file_path
        gt_file_path = file_path.replace('.mat', '_gt.png')
        #Load mat file
        data = scipy.io.loadmat(mat_file_path)
        wavelengths = data['wavelengths']
        hypercube = data['hyperfile']
        #select only the targeted wl_indices
        if self.wl_indices is not None:
            hypercube = hypercube[:,:,self.wl_indices]
            wavelengths = wavelengths[:,self.wl_indices]

        # transpose the hypercube to fit with the target image shape
        hypercube = hypercube.transpose(1,0,2)  

        #load ground truth
        gt = skimage.io.imread(gt_file_path)[:,:,1]
        
        #convert GT to one hot vector using the class indices
        target = np.zeros((gt.shape[0],gt.shape[1],len(self.class_indices)))
        for i,dataset_idx in enumerate(self.class_indices):
            target[gt==dataset_idx,i]=1.0

        return  {'id':file_id, 'wavelengths':wavelengths, 'hypercube':hypercube, 'target':target}

    def __getitem__(self, idx):

        real_idx = idx % self.image_list.shape[0]
        # Load the image
        if self.loaded_dataset_images is not None:
            image = self.loaded_dataset_images[real_idx]
        else:
            image = self._load_file(self.image_list[real_idx])

        image_id = image['id']
        image_x = image['hypercube']
        image_y = image['target']
        wavelengths = image['wavelengths']
       
        # perform albumentations
        if self.transforms is not None:
            transformed = self.transforms(image=image_x, mask=image_y)
            image_x = transformed['image']
            image_y = transformed['mask']
        
        # transform into appropiate datatype (float32)
        image_x = torch.as_tensor(image_x, dtype=torch.float32)
        image_y = torch.as_tensor(image_y, dtype=torch.float32)
        wavelengths = torch.as_tensor(wavelengths, dtype=torch.float32)
          
        sample_dict = {
            'id': image_id,
            'x': image_x,
            'y': image_y,
            'wavelengths':wavelengths}

        return sample_dict
    
    def __len__(self):
        return self.image_list.shape[0]*self.num_image_draws_per_epoch

    def get_class_weights(self, balancing_method: BalancingMethods, num_draws = 100):
        abundances = numpy.ones((len(self.classes),),dtype=float)
        num_current_draws = 0


        while num_current_draws < num_draws:
            for idx in range(self.image_list.shape[0]):
                data = self.__getitem__(idx)
                id = data['id']
                x = data['x']
                y = data['y']
                
                image_abundances = torch.sum(torch.sum(y,axis=-1),axis=-1)
                image_abundances_norm = image_abundances / torch.sum(image_abundances)
                
                abundances += image_abundances_norm.detach().numpy()
                num_current_draws += 1
                # print (num_current_draws)

        # normalize abundances
        class_abundances = abundances / numpy.sum(abundances)

        if balancing_method == BalancingMethods.no_balancing:
            class_weights = numpy.ones((len(self.category_list),),dtype=float)
        elif balancing_method == BalancingMethods.ignore_last_class:
            class_weights = numpy.ones((len(self.category_list),),dtype=float)
            class_weights[0] = 0
        elif balancing_method == BalancingMethods.substract_abundances:
            class_weights = 1 - class_abundances
        elif balancing_method == BalancingMethods.invert_abundances:
            class_weights = 1 / class_abundances
            class_weights = class_weights / numpy.sum(class_weights)
        elif balancing_method == BalancingMethods.effective_number_samples:
            effective_num = 1.0 - numpy.power(0.99, class_abundances)
            class_weights = (1.0 - 0.99) / effective_num
            class_weights = class_weights / numpy.sum(class_weights)
        elif balancing_method == BalancingMethods.decrease_background:
            class_weights = numpy.ones((len(self.category_list),),dtype=float)
            class_weights[0] = 0.01
        else:
            raise ValueError('Unknown balancing method: {}'.format(balancing_method))

        return class_abundances, class_weights
    
class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self, train_list, val_list, test_list,list_classes,class_indices,batch_size,train_transform,val_transform,test_transform, num_workers=6, load_in_memory=False, pin_memory=True, wl_indices=None, num_image_draws_per_epoch= 1):
        super().__init__()
        self.batch_size=batch_size
        self.train_transform=train_transform
        self.val_transform=val_transform
        self.test_transform = test_transform
        self.num_workers=num_workers
        self.pin_memory=pin_memory

        self.list_classes = list_classes  
        self.class_indices = class_indices
        self.wl_indices = wl_indices

        self.train_list = train_list
        self.val_list = val_list
        self.test_list = test_list

        self.load_in_memory = load_in_memory

        self.num_image_draws_per_epoch = num_image_draws_per_epoch
    
    def setup(self,stage = None):
        # Split dataset into training, validation and test sets
        self.shuffle_train = True
        self.train_set = SegmentationDataset(
                image_list = self.train_list, 
                list_classes = self.list_classes, 
                class_indices = self.class_indices,
                transforms=self.train_transform,
                load_in_memory = self.load_in_memory,
                wl_indices = self.wl_indices,
                num_image_draws_per_epoch=self.num_image_draws_per_epoch)

        self.val_set = SegmentationDataset(
                image_list = self.val_list, 
                list_classes = self.list_classes, 
                class_indices = self.class_indices,
                transforms=self.val_transform,
                load_in_memory = self.load_in_memory,
                wl_indices = self.wl_indices,
                num_image_draws_per_epoch=self.num_image_draws_per_epoch)

        self.test_set = SegmentationDataset(
                image_list = self.test_list, 
                list_classes = self.list_classes, 
                class_indices = self.class_indices,
                transforms=self.test_transform,
                load_in_memory = self.load_in_memory,
                wl_indices = self.wl_indices,
                num_image_draws_per_epoch=self.num_image_draws_per_epoch)

    def train_dataloader(self):
        train_loader=DataLoader(self.train_set, batch_size=self.batch_size,shuffle=self.shuffle_train,num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
        return train_loader
    
    def val_dataloader(self):
        val_loader=DataLoader(self.val_set, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
        return val_loader
    
    def test_dataloader(self):
        test_loader=DataLoader(self.test_set, batch_size=self.batch_size,shuffle=False,num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=False)
        return test_loader
    
if __name__ == '__main__':
    from core.albumentations_transforms import get_transforms
    # TENEMOS QUE INCLUIR DATA BALANCING!!!
    params = load_params()
    # TESTING DATASET
    num_workers = params['accelerator']['dataloader_cpu_workers']
    pin_memory = params['accelerator']['dataloader_pin_memory']
    batch_size = params['train']['batch_size']
    train_files = params['dataset']['train_files']
    val_files = params['dataset']['val_files']
    test_files = params['dataset']['test_files']
    
    load_all_data_in_memory = params['accelerator']['load_all_data_in_memory']
    num_image_draws_per_epoch = params['train']['num_image_draws_per_epoch']
    

    list_classes = []
    class_indices = []
    for class_dict in params['dataset']['classes']:
        list_classes.append(list(class_dict.keys())[0])
        class_indices.append(list(class_dict.values())[0])

    transforms_train = get_transforms(is_train=True,random_crop=True, input_size=params['model']['tile_size'], mean=params['preprocessing']['mean'], std=params['preprocessing']['std'])
    transforms_val = get_transforms(is_train=False,random_crop=True, input_size=params['model']['tile_size'], mean=params['preprocessing']['mean'], std=params['preprocessing']['std'])
    transforms_test = get_transforms(is_train=False,random_crop=False, input_size=params['model']['tile_size'], mean=params['preprocessing']['mean'], std=params['preprocessing']['std'])
    
    if params['model']['use_only_target_bands']:
        wl_indices = params['model']['target_bands']
    else:
        wl_indices = None
    ## Testing dataset
    #dataset train
    dataset = SegmentationDataset(
        image_list = train_files, 
        list_classes = list_classes, 
        class_indices = class_indices,
        transforms=transforms_train,
        load_in_memory = True,
        wl_indices = wl_indices,
        num_image_draws_per_epoch = num_image_draws_per_epoch)
    # dataset val
    dataset_val = SegmentationDataset(
        image_list = val_files, 
        list_classes = list_classes, 
        class_indices = class_indices,
        transforms=transforms_val,
        load_in_memory = True,
        wl_indices = wl_indices,
        num_image_draws_per_epoch = num_image_draws_per_epoch)
    
    for i in range(len(dataset)):
        sample = dataset[i]
        print(i, sample['x'].shape, sample['y'].shape)
        if i == 3:
            break  
    for i in range(len(dataset_val)):
        sample = dataset_val[i]
        print(i, sample['x'].shape, sample['y'].shape)
        if i == 3:
            break 
    # TESTING DATALOADER
    data_module = SegmentationDataModule(
        train_list = train_files, 
        val_list = val_files, 
        test_list = test_files,
        list_classes = list_classes,
        class_indices = class_indices,
        batch_size = batch_size,
        train_transform = transforms_train,
        val_transform = transforms_val,
        test_transform = transforms_test,
        num_workers = num_workers,
        pin_memory = pin_memory,
        wl_indices = wl_indices,
        load_in_memory = load_all_data_in_memory,
        num_image_draws_per_epoch = num_image_draws_per_epoch
    )
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    test_loader = data_module.test_dataloader()
    i= 0
    for batch in val_loader:
        for el in batch['id']:
            print(i, el)
            i += 1

    ## TESTING DATA BALANCING
    abundances, class_weights = dataset.get_class_weights(
        balancing_method=BalancingMethods.invert_abundances,
        num_draws=20)
    print(abundances)
    print(class_weights)