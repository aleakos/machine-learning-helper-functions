# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 12:40:17 2022

@author: owenpaetkau

Goal of this piece of code is to trim all of the images to appropriate size.
This trimming is done based on HN_002 image.

"""

import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import time

import pydicom
import os
from glob import glob

from dicomMethods import *


if __name__ == "__main__":
    wd = 'H:/HN_TransferLearning/2_output/05_dose_to_image/'
    wd_dose = 'H:/HN_TransferLearning/2_output/05_dose_to_image/dose/' 
    wd_ct = 'H:/HN_TransferLearning/2_output/05_dose_to_image/ct/'
    
    output = 'H:/HN_TransferLearning/2_output/06_crop_images/'
    output_ct = 'H:/HN_TransferLearning/2_output/06_crop_images/ct/'
    output_dose = 'H:/HN_TransferLearning/2_output/06_crop_images/dose/'

    reg_shift = pd.read_excel('H:/HN_TransferLearning/0_data/registration/RegistrationShifts.xlsx')
    patient_list = list(np.unique(reg_shift.Patient))
    
    for id, hn_id in enumerate(patient_list):
        print(f'Processing patient {hn_id}...')
        
        dose_file = f'dose_img_{hn_id}.npy'
        ct_file = f'ct_img_{hn_id}.npy'
        
        if (os.path.exists(output_ct + ct_file)):
            print(f"{hn_id} has already been processed.")
            continue        
        
        # Load the output from 05_Dose_to_Image.py.
        print(f'...importing images.')
        ct_img, dose_img = load_images(hn_id, wd, plot = False)
        
        # Crop the image to the appropriate size.
        print(f'...cropping images.')
        crop_size = [(150,450),(135,435),(212,512)]
        ct_img = crop_image(ct_img, crop_size)
        dose_img = crop_image(dose_img, crop_size)
        
        # Window the images. 
        print(f'...windowing images.')
        ct_img = window_image(ct_img)
        dose_img = window_image(dose_img)
        
        # Save output files.
        np.save(output_dose + dose_file, dose_img)
        np.save(output_ct + ct_file, ct_img)
        
        
        