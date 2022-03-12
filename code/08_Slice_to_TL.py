# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 15:57:36 2022

@author: owenpaetkau

Made a mistake, data wasn't ready to be pushed through. Need to place them
into the correct array shape so images have 3 channels.

"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time

import pydicom
import os
from glob import glob

from PIL import Image

from dicomMethods import *

from PIL import Image


def combine_channels(wd, slice_type, slice_num):
    ct = np.load(f'{wd}ct/ct_{slice_type}_{slice_num}.npy')
    dose = np.load(f'{wd}dose/dose_{slice_type}_{slice_num}.npy')
    
    full_list = []
    
    for ii in range(ct.shape[0]):
        lst = []
        
        lst.append(scale_image(ct[ii])) #CT
        lst.append(scale_image(dose[ii])) #Dose
        lst.append(scale_image(dose[ii] + ct[ii])) #CT+Dose
        
        lst_trans = np.array(lst).transpose()
        full_list.append(lst_trans)

    full_array = np.array(full_list)
    
    return full_array        
    

if __name__ == "__main__":
    
    wd = 'H:/HN_TransferLearning/2_output/07_slice_images/'  
    output = 'H:/HN_TransferLearning/2_output/08_images_to_TL/'

    sag_slices = np.arange(145, 156, 1)
    cor_slices = np.arange(115, 126, 1)
    axial_slices = np.arange(115, 146, 3)
    
    for sag, cor, axial in zip(sag_slices, cor_slices, axial_slices):
        print(f'Processing slices {sag} {cor} {axial}...') 
        
        # Combine CT, dose and ct+dose as channels.
        sag_array = combine_channels(wd, 'sagittal', sag)
        cor_array = combine_channels(wd, 'coronal', cor)
        axial_array = combine_channels(wd, 'axial', axial)
           
        np.save(output + f"sagittal_set_{sag}.npy", sag_array)
        np.save(output + f"coronal_set_{cor}.npy", cor_array)
        np.save(output + f"axial_set_{axial}.npy", axial_array)


        
        
        
        
        
        
        
        
        
        
        
        
        