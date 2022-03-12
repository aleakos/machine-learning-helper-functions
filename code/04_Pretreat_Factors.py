# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 09:12:27 2022

@author: owenpaetkau

Use this code to pull in pre-treatment factors and return the following:
    Tumour site
    Gender
    Tumour stage (T and N)
    Alcohol status
    Smoking status
    MDADI_SUM score
    Categorized MDADI_SUM score    

"""

import numpy as np
import pydicom
import os
import matplotlib.pyplot as plt
from glob import glob
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import scipy.ndimage
from skimage import morphology
from skimage import measure
from skimage.transform import resize
from sklearn.cluster import KMeans
import scipy 
from scipy.ndimage import label, morphology, interpolation
import pandas as pd

input_path = 'H:/HN_TransferLearning/0_data/'
output_path = 'H:/HN_TransferLearning/2_output/04_pretreat_results/'

df = pd.read_excel(input_path + 'pro_data_133pts.xlsx')

# Pull out easy information.
np.save(output_path + 'hn_id.npy',list(df.QoLID))
np.save(output_path + 'cancer_site.npy',list(df.CancerSite))
np.save(output_path + 'gender.npy',list(df.Gender))
np.save(output_path + 't_stage.npy',list(df.Tstage))
np.save(output_path + 'n_stage.npy',list(df.Nstage))
np.save(output_path + 'alcohol_intake.npy',list(df.AlcoholIntake))
np.save(output_path + 'smoking_history.npy',list(df.SmokingHistory))

# Create key for categorizing MDADI information.
col = 'MDADI_TOTAL_SUM'
conditions = [(df[col] >= 0) & (df[col] < 20),
    (df[col] >= 20) & (df[col] < 40),
    (df[col] >= 40) & (df[col] < 60),
    (df[col] >= 60) & (df[col] < 80),
    (df[col] > 80)]

# create a list of the values we want to assign for each condition
values = ['none', 'mild', 'moderate', 'severe','profound']

# create a new column and use np.select to assign values to it using our lists as arguments
mdadi_cat = np.select(conditions, values)
np.save(output_path + 'mdadi_labels.npy', list(mdadi_cat))




