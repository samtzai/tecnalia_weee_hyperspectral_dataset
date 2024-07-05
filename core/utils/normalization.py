from sklearn import preprocessing

def normalize(norm,input_data,wavelengths=None,n_wavelenght=394.451):
    #Normalize data with selected strategy
    if norm=='None':
        data_norm=input_data

    elif norm=='SNV':
        mean=input_data.mean().item()
        std=input_data.std().item()
        data_norm = (input_data - mean) / std

    elif norm== 'Total_int':
        sum_intensities=sum(input_data)
        data_norm=input_data/sum_intensities

    elif norm== 'Unit_norm':
        input_data=input_data.unsqueeze(0)
        data_norm=preprocessing.normalize(input_data, norm='l2')
            
    elif norm== 'Max_int':
        max_int=input_data.max().item()
        data_norm=input_data/max_int
            
    elif norm== 'Ref_line':
        idx=((wavelengths== n_wavelenght).nonzero(as_tuple=True)[0])
        # idx=((wavelengths== n_wavelenght).nonzero(as_tuple=True)[0][0]) #Data 1 spectrometer
        norm_intensity=input_data[idx].item()
        data_norm=input_data/norm_intensity

    input_data=data_norm     
    if norm!= 'Unit_norm':       
        input_data=input_data.unsqueeze(0)

    return input_data

