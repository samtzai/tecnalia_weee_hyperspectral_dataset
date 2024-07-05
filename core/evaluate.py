import set_python_path
from core.simple_models import EncoderDecoder, SpectralEncoderDecoder, UNet


import matplotlib
matplotlib.use('Agg')
import argparse
import os
import torch
import json
import pandas as pd
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from core.albumentations_transforms import get_transforms
from core.datamodule import  SegmentationDataModule
from core.utils.io import load_params,read_json
from core.utils.devices import config_device
from pytorch_lightning import seed_everything
import numpy as np
import cv2
import random

def generate_colors(label_list):
    # Obtener la lista de colores posibles
    list_colors = list(matplotlib.colors.CSS4_COLORS.values())
    # Obtener un color aleatorio
    random_color_name = random.choices(list_colors, k=len(label_list))
    colors = []
    for color in random_color_name:
        colors.append(tuple(int(c * 255) for c in matplotlib.colors.to_rgb(color)))
    return colors

def create_gt_mask(img, y_true_masks, y_pred_masks, img_dir, list_labels, colors, alpha=0.5):
    # ids = batch['id']
    
    # Obtener el cuarto canal de las imágenes x como base
    channels = img.shape[0]
    # get three equispeciated channels
    plot_channels = (2*channels//3, channels//2,  channels//3)
    base_images = img[plot_channels, :, :]
    # masks = batch['y']
    
    # Obtener las máscaras y aplicarlas a las imágenes base
    # for i in range(base_images.shape[0]):
    base_image = base_images.permute(1, 2, 0).cpu().numpy()  # Cambiar dimensiones y convertir a NumPy

    # Normalizar la imagen base si es de tipo flotante
    if base_image.dtype == np.float32 or base_image.dtype == np.float64:
        base_image = (base_image - base_image.min()) / (base_image.max() - base_image.min())
        
    # Convertir la imagen base a tipo entero si es de tipo flotante
    if base_image.dtype == np.float32 or base_image.dtype == np.float64:
        base_image = (base_image * 255).astype(np.uint8)

    fig, axs = plt.subplots(2, len(list_labels) + 1, figsize=(25, 5))
        
    # Mostrar la imagen base en la primera subtrama
    for i in range(2):
        axs[i, 0].imshow(base_image)
        axs[i, 0].set_title('Imagen base')
        axs[i, 0].axis('off')
        
    # Aplicar la máscara a la imagen base y mostrarla
    for i in range(len(list_labels)):
        masked_image_gt = base_image.copy()
        masked_image_pred = base_image.copy()
        masked_image_gt[y_true_masks != i] = [100, 100, 100]
        masked_image_pred[y_pred_masks != i] = [100, 100, 100]

        axs[0, i + 1].imshow(masked_image_gt)
        axs[1, i + 1].imshow(masked_image_pred)
        axs[0, i + 1].set_title(list_labels[i])
        axs[0, i + 1].axis('off')
        axs[1, i + 1].axis('off')
    
    # Ajustar el diseño de la figura
    plt.tight_layout()
    # Guardar la figura como imagen
    plt.savefig(img_dir + '/image_masks.png')
    plt.close(fig)  # Cerrar la figura para liberar memoria

        # ######################################################################
    gt_base = base_image.copy()
    pred_base = base_image.copy()
    for i in range(len(list_labels)):
        color = colors[i]
        for c in range(3):
            gt_base[:, :, c] = np.where(y_true_masks == i, gt_base[:, :, c]*(1 - alpha) + alpha*color[c], gt_base[:, :, c])
        for c in range(3):
            pred_base[:, :, c] = np.where(y_pred_masks == i, pred_base[:, :, c]*(1 - alpha) + alpha*color[c], pred_base[:, :, c])
    cv2.imwrite(img_dir + '/color_gt_mask_image.png', gt_base)
    cv2.imwrite(img_dir + '/color_pred_mask_image.png', pred_base)

eps=0.000001
def extract_metrics(confusion_matrix, list_classes, eps=1e-10):
    num_classes = confusion_matrix.shape[0]
    metrics = {}
    global_metrics = {
        "Precision": 0,
        "Recall": 0,
        "F1 Score": 0,
        "Accuracy": 0,
        "IoU": 0
    }

    for cls in range(num_classes):
        TP = confusion_matrix[cls, cls]
        FN = confusion_matrix[cls].sum() - TP
        FP = confusion_matrix[:, cls].sum() - TP
        TN = confusion_matrix.sum() - TP - FN - FP

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * (precision * recall) / (precision + recall + eps)
        accuracy = (TP + TN) / confusion_matrix.sum()

        # Cálculo del IoU (Índice de Jaccard)
        iou = TP / (TP + FP + FN + eps)

        metrics[f"Class {cls}"] = {
            "name": list_classes[cls],
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1,
            "Accuracy": accuracy,
            "IoU": iou
        }

        # Actualizar métricas globales
        global_metrics["Precision"] += precision
        global_metrics["Recall"] += recall
        global_metrics["F1 Score"] += f1
        global_metrics["Accuracy"] += accuracy
        global_metrics["IoU"] += iou

    # Calcular la media de las métricas globales
    num_samples = len(list_classes)
    for metric in global_metrics:
        global_metrics[metric] /= num_samples

    metrics["Global Metrics"] = global_metrics

    return metrics
   
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='test')
    parser.add_argument('--params', help='params file', default='params.yaml')
    parser.add_argument('--subset', help='required subset, train, val or test', required=True)

    args = parser.parse_args()
    params = load_params(args.params)

    # disable gradients
    torch.set_grad_enabled(False)

    torch_accelerator = params['accelerator']['torch_accelerator']
    torch_devices = params['accelerator']['torch_devices']

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


    output_folder = os.path.join('outputs',params['experiment_name'], 'evaluate', args.subset)
    model_folder = os.path.join('outputs',params['experiment_name'], 'train')
    os.makedirs(output_folder, exist_ok=True)





    """Set the random seed so all of our experiments can be reproduced"""
    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED
    seed_everything(42, workers=True)

    #Load Model
    model_info = read_json(model_folder + '/model.json')

    model_name = model_info['backbone_info']['model_name']
    input_size = model_info['input_size']
    num_channels =  model_info['num_channels']
    activation_name = model_info['activation_name']
    num_classes =  model_info['num_classes']
    intermediate_layers = model_info['intermediate_layers']

    model_mean =  model_info['backbone_info']['model_mean']
    model_std =  model_info['backbone_info']['model_std']

    #reduce the number of bands if needed
    if params['model']['use_only_target_bands']:
        wl_indices = params['model']['target_bands']
    else:
        wl_indices = None



    model_type = params['model']['model_type']
    if model_type == 'encoderdecoder':
        model = EncoderDecoder(num_channels,num_classes,activation_name=activation_name)
    elif model_type == 'unet':
        model = UNet(num_channels,num_classes,activation_name=activation_name)
    elif model_type == 'spectral':
        model = SpectralEncoderDecoder(num_channels,num_classes,activation_name=activation_name)
    else:
        raise ValueError('Model type %s not recognized' % model_type)



    model_weights = torch.load(model_folder + '/model.pt')
    model.load_state_dict(model_weights,strict=True)

    # SEND MODEL TO DEVICE
    model.to(torch_devices[0])
    model.eval()

    batch_size = 1

    
    transforms_train = get_transforms(is_train=False,random_crop=False, input_size=params['model']['tile_size'],  mean=model_mean, std=model_std,max_pixel_value=1.0)
    transforms_val = get_transforms(is_train=False,random_crop=False, input_size=params['model']['tile_size'],  mean=model_mean, std=model_std,max_pixel_value=1.0)
    transforms_test = get_transforms(is_train=False,random_crop=False, input_size=params['model']['tile_size'],  mean=model_mean, std=model_std,max_pixel_value=1.0)


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
        num_image_draws_per_epoch = 1
    )
    data.setup()

    
    # Perform evaluation

    os.makedirs(output_folder + '/images', exist_ok=True)
    if args.subset == 'train':
        data_iterator = data.train_dataloader().__iter__()
    elif args.subset == 'val':
        data_iterator = data.val_dataloader().__iter__()    
    elif  args.subset == 'test':
        data_iterator = data.test_dataloader().__iter__()
    
    colors = generate_colors(list_classes)
    
    global_confussion_matrix = None
    num_el=0
    total_el = len(data_iterator)

    thr_s1=[]
    thr_s2=[]

    for batch in data_iterator:

        split1={'gt':[],'pred':[]}
        split2={'gt':[],'pred':[]}
        
        print('evaluating batch %d/%d' % (num_el,total_el ))
        id= batch['id']
        x= batch['x'].to(torch_devices[0])
        y= batch['y']


        # if num_el>10:
        #     break
        num_el+=1

        with torch.inference_mode():
            logits=model.forward(x,apply_activation = True)

        y_pred_batch = torch.argmax(logits,dim=1).detach().cpu().numpy()
        y_true_batch = torch.argmax(y,dim=1).detach().cpu().numpy()

        for i in range(y_pred_batch.shape[0]):

            # Create image masks
            if num_el == 0:
                image_dir = os.path.join(output_folder)
                os.makedirs(image_dir, exist_ok=True)
                create_gt_mask(batch['x'][i], y_true_batch[i], y_pred_batch[i], image_dir, list_classes, colors)
  
            image_dir = os.path.join(output_folder + '/images', id[i])
            os.makedirs(image_dir, exist_ok=True)
            create_gt_mask(batch['x'][i], y_true_batch[i], y_pred_batch[i], image_dir, list_classes, colors)


            # Analyse metrics
            y_pred = y_pred_batch[i].flatten()
            y_true = y_true_batch[i].flatten()

            gt=y[i].detach().cpu().unsqueeze(0).numpy()
            logits_pred=logits[i].detach().cpu().unsqueeze(0).numpy()

            if i<=(round(y_pred_batch.shape[0]/2)-1):
               
                if len(split1['gt'])!=0:
                    split1['gt']=np.concatenate((split1['gt'],gt))
                else:
                    split1['gt']=gt
                if len(split1['pred'])!=0:
                    split1['pred']=np.concatenate((split1['pred'],logits_pred))
                else:
                    split1['pred']=logits_pred

            else:
                if len(split2['gt'])!=0:
                    split2['gt']=np.concatenate((split2['gt'],gt))
                else:
                    split2['gt']=gt
                if len(split2['pred'])!=0:
                    split2['pred']=np.concatenate((split2['pred'],logits_pred))
                else:
                    split2['pred']=logits_pred
                    
            # Calculate image confussion matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(len(list_classes)))
            if global_confussion_matrix is None:
                global_confussion_matrix = cm
            else:
                global_confussion_matrix += cm

            # save image confussion matrix in csv
            df = pd.DataFrame(cm, index=list_classes, columns=list_classes)
            
            df.to_csv(image_dir + '/' + id[i].replace('/','_') + '.csv')
            
  

            
        
    #normalize global confussion matrix
    global_confussion_matrix_norm = global_confussion_matrix.astype('float') / global_confussion_matrix.sum(axis=1)[:, np.newaxis]
    
        
    #store global confussion matrix  as csv
    df = pd.DataFrame(global_confussion_matrix_norm, index=list_classes, columns=list_classes)
    df.to_csv(output_folder + '/confussion_matrix_norm.csv')
    #plot confussion matrix with labels
    plt.figure(figsize=(32,32))
    disp = ConfusionMatrixDisplay(confusion_matrix=global_confussion_matrix_norm,display_labels=list_classes)
    disp.plot()
    plt.savefig(output_folder + '/confussion_matrix_norm.png')
    plt.close()



    #store global confussion matrix  as csv
    df = pd.DataFrame(global_confussion_matrix, index=list_classes, columns=list_classes)
    df.to_csv(output_folder + '/confussion_matrix.csv')
    #plot confussion matrix with labels
    plt.figure(figsize=(32,32))
    disp = ConfusionMatrixDisplay(confusion_matrix=global_confussion_matrix,display_labels=list_classes)
    disp.plot()
    plt.savefig(output_folder + '/confussion_matrix.png')
    plt.close()


    #Calculate metrics from confusion matrix by class and save to json
    metrics = extract_metrics(global_confussion_matrix, list_classes)

    with open(output_folder + '/metrics.json', 'w+') as metrics_file:
        json.dump(metrics, metrics_file, indent=4)
    
    