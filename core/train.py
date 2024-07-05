import math
import set_python_path
from core.albumentations_transforms import get_transforms
from core.callbacks import get_callbacks_list
import argparse
import os
import pytorch_lightning as pl
import torch
import pandas as pd
import shutil
import matplotlib.pyplot as plt
from core.datamodule import BalancingMethods, SegmentationDataModule, SegmentationDataset
from core.pl_model import pl_model
from core.utils.io import load_params,read_json,write_json
from core.utils.devices import config_device
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import seed_everything
from pytorch_lightning.strategies import DDPStrategy

def get_trainer(params, devices,accelerator,logger,callbacks_list,  strategy='ddp'):
    trainer=pl.Trainer(max_epochs=params['train']['epochs'], 
                         accelerator=accelerator, #specify the accelerator to use
                         devices=devices, #Select the devices to use
                         logger=logger, 
                         strategy=DDPStrategy() if strategy=='ddp' else "Auto",
                         callbacks=callbacks_list,
                         deterministic=False, # To ensure reproducibility
                         precision=16
                        #  default_root_dir=check_point_path,
                        #  log_every_n_steps=1,
                         )
    return trainer

def get_class_weights(train_files,list_classes,class_indices,transforms_train,load_in_memory,wl_indices, loss_weight_calculation, draw_images):

    dataset = SegmentationDataset(
        image_list = train_files, 
        list_classes = list_classes, 
        class_indices = class_indices,
        transforms=transforms_train,
        load_in_memory = load_in_memory,
        wl_indices = wl_indices)
    balancing_method = BalancingMethods.from_string(loss_weight_calculation)
    abundances, weights = dataset.get_class_weights(
        balancing_method,
        draw_images)
    return abundances, weights


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--params', help='params file', default='params.yaml')

    args = parser.parse_args()
    params = load_params(args.params)

    output_folder  = os.path.join('outputs', params['experiment_name'], 'train')
    os.makedirs(output_folder, exist_ok=True)

    """Set the random seed so all of our experiments can be reproduced"""
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED
    seed_everything(params['random_seed'], workers=True)

    # LOAD PARAMS
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




    #reduce the number of bands if needed
    if params['model']['use_only_target_bands']:
        wl_indices = params['model']['target_bands']
    else:
        wl_indices = None



    # Balance dataset abundances by class  weight
    balancing_method = params['train']['loss_weight_calculation']
    loss_weight_calculation_draws = params['train']['loss_weight_calculation_draws']
    transforms_train_dummy = get_transforms(is_train=True,random_crop=True, input_size=params['model']['tile_size'])
    
    class_abundances, class_weights = get_class_weights(
        train_files = train_files,
        list_classes = list_classes,
        class_indices = class_indices,
        transforms_train = transforms_train_dummy,
        load_in_memory = load_all_data_in_memory,
        wl_indices = wl_indices, 
        loss_weight_calculation = balancing_method, 
        draw_images = loss_weight_calculation_draws
    )

    # CREATE THE MODEL
    model = pl_model(params,class_weights)
    model_card = model.get_model_card()

    # get transforms
    transforms_train = get_transforms(is_train=True,random_crop=True, input_size=params['model']['tile_size'],  mean=model_card['model_mean'], std=model_card['model_std'],max_pixel_value=1.0)
    transforms_val = get_transforms(is_train=False,random_crop=True, input_size=params['model']['tile_size'], mean=model_card['model_mean'], std=model_card['model_std'],max_pixel_value=1.0)
    transforms_test = get_transforms(is_train=False,random_crop=False, input_size=params['model']['tile_size'],  mean=model_card['model_mean'], std=model_card['model_std'],max_pixel_value=1.0)
    
    data=SegmentationDataModule(
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
    data.setup()

    # CALLBACKS
    callbacks_list= get_callbacks_list(params, output_folder)
    # END CALLBACKS

    #LOGGER
    # Save the training process metrics in a csv
    #Erase logger files if they exist
    logger = CSVLogger(output_folder,version=0)

    #TRAINER
    # trainer = get_trainer(params,"auto","auto",logger,callbacks_list, "auto")
    check_point_path = os.path.join(output_folder, 'last.ckpt')

    torch_accelerator = params['accelerator']['torch_accelerator']
    torch_devices = params['accelerator']['torch_devices']
    trainer = get_trainer(params, torch_devices,torch_accelerator,logger,callbacks_list)

    # RUN TRAINING
    if params['train']['resume_training']:
        if os.path.exists(check_point_path):
            ckpt_path=check_point_path
        else:
            ckpt_path = None
    else:
        ckpt_path = None        
            
    trainer.fit(model=model, datamodule=data, ckpt_path=ckpt_path)

    #If there is no callback to save the best weights, save checkpoint of the state of your last training epoch
    try:
        if not(params['train']['restore_best_weights']):
            trainer.save_checkpoint(output_folder+"/weights.ckpt") 
            ckpt= torch.load(output_folder+'/weights.ckpt') #Load the saved checkpoint
        else:
            weights_filename=callbacks_list[0].best_model_path
            ckpt= torch.load(weights_filename) #Load the saved checkpoint
    except:
        trainer.save_checkpoint(output_folder+"/weights.ckpt") 
        ckpt= torch.load(output_folder+'/weights.ckpt') #Load the saved checkpoint
    
    model.load_state_dict(ckpt['state_dict'])
    torch.save(model.model.state_dict(),os.path.join(output_folder,'model.pt')) #Save the model only with the weights
    # END RUN TRAINING

    try:
        # MODIFY LOGGING CSV STRUCTURE
        # read the csv file
        df = pd.read_csv(os.path.join(output_folder, 'lightning_logs', "version_0", "metrics.csv"))
        #Group together values of each epoch
        df= df.groupby(['epoch'], sort=False).sum()
        # writing into the file
        df.to_csv(os.path.join(output_folder, 'lightning_logs', "version_0", "metrics.csv"))


        # DRAW TRAINING PROGRESSION PLOT
        csv_path = os.path.join(output_folder, 'training.csv')
        shutil.copy(os.path.join(output_folder, 'lightning_logs', "version_0", "metrics.csv"), csv_path)
        shutil.rmtree(os.path.join(output_folder, 'lightning_logs'))
        df_training = pd.read_csv(csv_path)
        df_loss = df_training[df_training["train_loss"] > 0]
        df_val_loss = df_training[df_training["val_loss"] > 0]
        plt.plot(df_loss['epoch'], df_loss['train_loss'], label='loss')
        plt.plot(df_val_loss['epoch'], df_val_loss['val_loss'], label='val_loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig(os.path.join(output_folder, 'training.png'))

    except:
        pass

    # Save model information json
    model_info = dict()
    model_info['list_classes'] = model.list_classes
    model_info['num_classes'] = len(model.list_classes)
    model_info['task']="semantic_segmentation"
    model_info['input_size'] = model.image_size
    model_info['num_channels'] = model.num_channels
    model_info['backbone_info'] = model.get_model_card()
    model_info['intermediate_layers'] = model.intermediate_layers

    model_info['activation_name'] = model.activation
    write_json(os.path.join(output_folder, 'model.json'), model_info)

    # save params json
    write_json(os.path.join(output_folder, 'params.json'), params)
  








