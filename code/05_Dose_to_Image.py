# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:52:48 2022

@author: owenpaetkau
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

    wd_dose = 'H:/HN_TransferLearning/0_data/dose/'   
    wd_ct = 'H:/HN_TransferLearning/0_data/ct/'
    
    output_dose = 'H:/HN_TransferLearning/2_output/05_dose_to_image/dose/'
    output_ct = 'H:/HN_TransferLearning/2_output/05_dose_to_image/ct/'
    
    reg_shift = pd.read_excel('H:/HN_TransferLearning/0_data/registration/RegistrationShifts.xlsx')
    patient_list = list(np.unique(reg_shift.Patient))
    
    baseline = np.array([-300, -236, -583]) # Taken from first slice of HN_002.
    
    for id, hn_id in enumerate(patient_list): 
        print(f'Processing patient {hn_id}...')
        start = time.time()
        
        dose_file = f'dose_image_{hn_id}.npy'
        ct_file = f'ct_image_{hn_id}.npy'
        
        # Check to see if the output file exists.
        if (os.path.exists(output_ct + ct_file)):
            print(f"{hn_id} has already been processed.")
            continue

        # Define the deformation shifts.
        deformation = np.array((reg_shift[reg_shift.Patient == hn_id].X.values[0],
                                reg_shift[reg_shift.Patient == hn_id].Y.values[0],
                                reg_shift[reg_shift.Patient == hn_id].Z.values[0]))

        #-------------------------------------------------------------------------
        # Import and apply operations to files!
        #   1) Import ct + dose.
        #   2) Resample to 1 mm^3 voxel size.
        #   3) Resize image to [512,512,512] by cropping or padding end or array.
        #   4) Apply registration .ImagePositionPatient shifts.
        #-------------------------------------------------------------------------
        
        # Load ct and dose file.
        print(f'...importing files.')
        ct = load_scan(wd_ct + f'{hn_id}/') # This is [Z, Y, X]
        dose_arr, dose = load_dose(wd_dose + f'{hn_id}/') # This is [Z,Y,X]
    
        print(f'...imported {len(dose)} dose file(s) and {len(ct)} CT slices.')    
        
        # Pull out voxel information from dose and ct files.
        dose_ps = dose[0].PixelSpacing
        dose_thick = dose[0].GridFrameOffsetVector[1] - dose[0].GridFrameOffsetVector[0] 
        
        ct_ps = ct[0].PixelSpacing
        ct_thick = ct[0].SliceThickness
        
        # Pull out array from scans and dose file. 
        # Need to swap from [Z,Y,X] to [X,Y,Z].
        ct_img = np.swapaxes(get_pixels_hu(ct),0,-1)
        dose_arr = np.swapaxes(dose_arr, 0, -1)
        
        # Re-sample the images to a 1 mm^3 voxel size.
        print(f'...resampling image.')
        dose_img = resample(dose_arr, dose_thick, dose_ps)
        ct_img = resample(ct_img, ct_thick, ct_ps)
        
        # Resize the images so they are a common size!
        print(f'...resizing image.')
        shape = [512, 512, 512]
        dose_img = resize_image(dose_img, crop = shape)
        ct_img = resize_image(ct_img, crop = shape)
        
        # Resize the images so they are a common size!
        print(f'...shifting image.')
        ct_pos = np.array(ct[0].ImagePositionPatient)
        dose_pos = np.array(dose[0].ImagePositionPatient)
        
        dose_to_ct = dose_pos - ct_pos
        align_shift = ct_pos - baseline
        
        dose_img = registration_shift(dose_img, dose_to_ct + align_shift, deformation)
        ct_img = registration_shift(ct_img, align_shift, deformation)
        

        #-------------------------------------------------------------------------
        # Print out the appropriate plots and save the files.
        #-------------------------------------------------------------------------
        
        # Save output files.
        np.save(output_dose + dose_file, dose_img)
        np.save(output_ct + ct_file, ct_img)
        
        end = time.time()
        print(f'Finished processing {hn_id} in {(end - start) / 60:.1f} minutes.')
        
        # Optional plotting of slices.

        plt.figure(1)   
        a3 = plt.subplot(2, 5, (id % 10) + 1)
        plt.imshow(ct_img[256,:,:],cmap='gray')
        plt.imshow(dose_img[256,:,:],cmap='jet', alpha = 0.5)
        
        plt.figure(2)   
        a3 = plt.subplot(2, 5, (id % 10) + 1)
        plt.imshow(ct_img[:, 256,:],cmap='gray')
        plt.imshow(dose_img[:, 256,:],cmap='jet', alpha = 0.5)
        
        plt.figure(3)   
        a3 = plt.subplot(2, 5, (id % 10) + 1)
        plt.imshow(ct_img[:, :, 256],cmap='gray')
        plt.imshow(dose_img[:, :, 256],cmap='jet', alpha = 0.5)

        























