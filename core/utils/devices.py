# -*- coding: utf-8 -*-
__author__ = 106360

import torch
import os

import subprocess as sp
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


def config_device(computing_device):
    """
    Valid with Pytorch Lightning
    This method selects computing in a particular computing device (i.e. a particular
    GPU or CPU) of your system.

    Parameters
    ----------
    computing_device: string
        The wanted computing device. Could be either 'gpu:X' or 'cpu:X' substituting the
        X for the ID number of the desired computing device (e.g. 'gpu:0').
    """
    if 'takethemall' in computing_device.lower():
        avail_gpu_devices = torch.cuda.device_count()
        gpu_devices=[]
        for pos in range(avail_gpu_devices):
            gpu_devices.append(torch.cuda.get_device_name(pos))

        print(f'Available physical devices: {gpu_devices}')
        device="-1"
        accelerator="gpu"

    elif 'find_a_free_gpu' in computing_device.lower():
        free_memory_list = get_gpu_memory()
        selected_device_idx = free_memory_list.index(max(free_memory_list))        
        selected_device ='/GPU:%d' % selected_device_idx
        print('Selecting device: %s' % selected_device)
        return config_device(selected_device)

    elif 'gpu' in computing_device.lower():
        device_number = int(computing_device.rsplit(':', 1)[1])
        
        # Select desired GPU
        #os.environ["CUDA_VISIBLE_DEVICES"]=str(device_number)
        
        device=[device_number]
        visible_gpus = torch.cuda.get_device_name(device_number)
        print('Selected devices: ' + str(visible_gpus))
        accelerator="gpu"
        

        

    elif 'cpu' in computing_device.lower():
        # Select CPU equivalent code for pl
        device=None
        accelerator="cpu"

    return device,accelerator

if __name__ == "__main__":
    pass
