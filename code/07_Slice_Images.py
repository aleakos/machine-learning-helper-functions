# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 10:53:09 2022

@author: owenpaetkau

Goal of this piece of code is to import the 3D array images from 05_Dose_to_image.py,
and compile arrays of single slices of interest in both the dose and ct scans.

Additionally, I want to be able to scale both of those outputs to be between 0 and 255
to be suitable as inputs for transfer learning algorithms.
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


if __name__ == "__main__":
    
    wd = 'H:/HN_TransferLearning/2_output/06_crop_images/'    
    output = 'H:/HN_TransferLearning/2_output/07_slice_images/'

    reg_shift = pd.read_excel('H:/HN_TransferLearning/0_data/registration/RegistrationShifts.xlsx')
    patient_list = list(np.unique(reg_shift.Patient))
    
    sag_slices = np.arange(145, 156, 1)
    cor_slices = np.arange(115, 126, 1)
    axial_slices = np.arange(115, 146, 3)
    
    for sag, cor, axial in zip(sag_slices, cor_slices, axial_slices):
        print(f'Processing slices {sag} {cor} {axial}...')       
    
        if (os.path.exists(f'{output}/dose/dose_axial_{axial}.npy')):
            print(f"Slices {sag} {cor} {axial} have been processed.")
            continue   
        
        start = time.time()
        
        sag_dose, cor_dose, axial_dose = [], [], []
        sag_ct, cor_ct, axial_ct = [], [], []
        
        for id, hn_id in enumerate(patient_list):
            
            # Load dose and ct. 
            print(f'...loading images for {hn_id}.')
            ct_img = np.load(wd + f'ct/ct_img_{hn_id}.npy')
            dose_img = np.load(wd + f'dose/dose_img_{hn_id}.npy')
            
            # For each array, slice along specific axis and save file.
            # Sagittal array:
            sag_ct.append(ct_img.take(indices = sag, axis = 0))
            sag_dose.append(dose_img.take(indices = sag, axis = 0))
            
            # Coronal array:
            cor_ct.append(ct_img.take(indices = cor, axis = 1))
            cor_dose.append(dose_img.take(indices = cor, axis = 1))
            
            # Axial array:
            axial_ct.append(ct_img.take(indices = axial, axis = 2))
            axial_dose.append(dose_img.take(indices = axial, axis = 2))
                       
            
        np.save(f'{output}/ct/ct_sagittal_{sag}.npy', sag_ct)
        np.save(f'{output}/dose/dose_sagittal_{sag}.npy', sag_dose)
        
        np.save(f'{output}/ct/ct_coronal_{cor}.npy', cor_ct)
        np.save(f'{output}/dose/dose_coronal_{cor}.npy', cor_dose)
        
        np.save(f'{output}/ct/ct_axial_{axial}.npy', axial_ct)
        np.save(f'{output}/dose/dose_axial_{axial}.npy', axial_dose)
        
        end = time.time()
        
        print(f'Finished process slices {sag} {cor} {axial} in {(end - start) / 60:.1f} minutes.')
            

            

            
            
            
            
        
        
        