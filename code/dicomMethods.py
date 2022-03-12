###############################################################################
### Code courtesy of Kailyn Stenhouse, adjusted by Owen Paetkau for  ##########
### HN ART applications.                                             ##########
### Received May 21st, 2021.                                         ##########
###
### Added code from Fletcher Barrett, and adjusted by Owen. This     ##########
### is code used for Fletcher's deep learning project.               ##########
### Edits made March 1st, 2022.
###############################################################################

###############################################################################
################################### IMPORTS ###################################
###############################################################################

#SYSTEM IMPORTS
import copy
import csv
import os
import glob

#DICOM PROCESSING IMPORTS
import pydicom
#from dicompylercore import dvh, dvhcalc, dicomparser

#DATA PROCESSING IMPORTS
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats as spstat
import scipy.spatial.distance as spdist
import scipy.interpolate as interp
from scipy.ndimage import label, morphology, interpolation
from collections import deque

#PLOTTING IMPORTS
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

###############################################################################
#################### 3D Scroller Overlay Class & Method #######################
###############################################################################

class IndexTracker(object):
    def __init__(self, ax, X, Y, axis):
        self.ax = ax
        self.X = X
        self.Y = Y
        self.axis = axis
        self.slices, _, _  = X.shape
        self.ind = self.slices // 2
        
        self.im1 = ax.imshow(self.X.take(indices = self.ind, axis = self.axis), cmap="gray")
        self.im2 = ax.imshow(self.Y.take(indices = self.ind, axis = self.axis), cmap="jet", alpha=.5)

        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self): 
        im1_data = self.im1.to_rgba(self.X.take(indices = self.ind, axis = self.axis), alpha=self.im1.get_alpha())
        im2_data = self.im2.to_rgba(self.Y.take(indices = self.ind, axis = self.axis), alpha=self.im2.get_alpha())

        self.im1.set_data(im1_data)
        self.im2.set_data(im2_data)

        self.ax.set_ylabel('slice %s' % self.ind)
        self.im1.axes.figure.canvas.draw()
        
def plot3d(image1, image2, axis = 2):
    '''
    Overlay two 3D arrays and index along any given axis. If you only want to view
    one image, place the same array in both images.

    Parameters
    ----------
    image1 : numpy.ndarray
        First image - usually ct due to colouring.
    image2 : numpy.ndarray
        Second image - usually dose due to colouring.
    axis : int, optional
        DESCRIPTION. The default is 2.

    Returns
    -------
    tracker : TYPE
        Need to save output so it remains active after command line closes.

    '''
    if axis > (len(image1.shape) - 1):
        axis = (len(image1.shape) - 1)
        print(f'Axis is out of bounds, it has been set to {axis}.')
    
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, image1, image2, axis)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    return tracker

def load_images(patient_id, path = 'H:/HN_TransferLearning/2_output/05_dose_to_image/', plot = True):
    ct_img = np.load(path + f'ct/ct_image_{patient_id}.npy')
    dose_img = np.load(path + f'dose/dose_image_{patient_id}.npy')    
    
    if plot == True :
        track = plot3d(ct_img, dose_img)
        return ct_img, dose_img, track
    else : 
        return ct_img, dose_img

###############################################################################
#################### Fletcher & Owens DICOM & DATA PROCESSING #################
###############################################################################

''' All of the following have been adjusted by Owen! '''

def load_scan(path):
    '''
    Load the CT slices from a directory. The directory must contain only
    CT slices!

    Parameters
    ----------
    path : string
        Directory leading to the CT files.

    Returns
    -------
    slices : list
        List of imported ct slices as pydicom format.

    '''
    slices = [pydicom.read_file(path + '/' + s) for s in os.listdir(path)]
    
    # Inverted the sorting to match the dose file.
    slices.sort(reverse = True, key = lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices

def load_dose(path):
    '''
    Load all of the dose files in a single path, and sum the dose arrays together.
    Biggest use is when a patient has VMAT arcs in their dose distribution.

    Parameters
    ----------
    path : string
        Path to the dose files.

    Returns
    -------
    dose_sum : numpy.ndarray
        Summed dose array after importing all of the dose files.
    dose_files : list
        List of the pydicom dose files imported.

    '''
    file_paths = glob.glob(path + 'RD.*')
    
    dose_sum = 0
    dose_files = []
    for file in file_paths:
        dose = pydicom.read_file(file)
        dose_files.append(dose)
        dose_sum += dose.pixel_array * dose.DoseGridScaling
        
    return dose_sum, dose_files


def get_pixels_hu(scans):
    '''
    Take in the list of scans from load_scan method and scale them using
    the HU values.

    Parameters
    ----------
    scans : list
        List of Pydicom scans from load_scan method.

    Returns
    -------
    TYPE : numpy.ndarray
        Output an array, scaled with HU units. 
        The shape of output is [Z,Y,X].

    '''
    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope
    
    if slope != 1:
        image = slope * image.astype(np.float32)
        image = image.astype(np.int16)
        
    image += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def resample(image, image_thickness, pixel_spacing): 
    '''
    Resampled 3D dose or ct image according to pixel spacing and slice
    thickness to project it into a 1 mm x 1 mm x 1 mm grid.

    Parameters
    ----------
    image : numpy.ndarray
        Dose file from pixel.array or imported ct scan after get_pixels_hu.
    image_thickness : pydicom.valuerep.DSfloat
        Slice thicknes of CT scans.
    pixel_spacing : pydicom.multival.MultiValue
        Spacing of pixels in [x,y] directions.

    Returns
    -------
    resampled_image : numpy.ndarray
        Resampled array in 1 mm x 1 mm x 1 mm grid.

    '''
    # Remade by Owen Feb 24th, 2022.
    x_pixel = float(pixel_spacing[0])
    y_pixel = float(pixel_spacing[1])
    
    size = np.array([x_pixel, y_pixel, float(image_thickness)])

    # Changed this from 2d interpolation to 3d interpolation.
    resampled_image = interpolation.zoom(image,size)
    
    return resampled_image

def resize_image(image, new_dim = [750,750,750], crop = [512,512,512]):
    '''
    Resize the dose or ct file to a common size, after resampling and 
    before applying registration shifts. Pads at end of the array
    so the top left pixel remains aligned so shifts are applied correctly.

    Parameters
    ----------
    image : numpy.ndarray
        Input 3D array, typically a dose or ct file.
    new_dim : TYPE, optional
        Temporary image pad size. The default is [750,750,750].
    crop : TYPE, optional
        Final image cropped size. The default is [512,512,512].

    Returns
    -------
    final_image : numpy.ndarray
        Resized dose or ct file.

    '''
    # Remade by Owen Feb 24th, 2022.   
    dim_dif = new_dim - np.array(image.shape)
    
    # Pad step to get to new_dim size.
    pad = ((0,dim_dif[0]),(0,dim_dif[1]),(0,dim_dif[2]))
    temp_image = np.pad(image,pad_width = pad, mode = 'constant', constant_values = 0)

    # Crop image down to cropped size.
    final_image = temp_image[:crop[0],:crop[1],:crop[2]]
    
    return final_image

def registration_shift(img,extra_shift,deformation):
    '''     
    Apply the translational shifts for all three axis. 
    
    This will wrap around if the shift exceeds the size of the array.
    No need to consider pixel_info as they have been resampled into 
    1mm x 1mm x 1mm grids using resample method.   
    
    Need the resize_image applied first. If not, shifts may wrap around.
    
    Other change was to swap Z_shift and X_shift operations. I think this
    works because I swapped the axis before registration, while
    Fletcher did this operation afterwards.

    Parameters
    ----------
    img : numpy.ndarray
        Dose or CT file after being processed by resize_image.
    extra_shift : numpy.ndarray
        Shift from the alignment of dose and ct files, and aligning
        the .ImagePositionPatient to reference.
    deformation : numpy.ndarray
        Apply the shifts from image registration.

    Returns
    -------
    numpy.ndarray
        CT or dose file with the appropriate shifts applied.

    '''
    X_shift =  int(np.round(deformation[0] + extra_shift[0]))
    Y_shift =  int(np.round(deformation[1] + extra_shift[1]))
    Z_shift =  int(np.round(deformation[2] + extra_shift[2]))
    #print (f'Shifted by {X_shift} {Y_shift} {Z_shift}.')
    
    l3 = []
    for k in range(len(img)):
        l1 = []
        for i in range(img[0].shape[0]):
            items = deque(img[k][i])
            items.rotate(Z_shift) # Swapped to Z from previous.
            l1.append(items)
        
        temp = np.array(l1, dtype = np.float32)
        l2 = []
        for j in range(img[0].shape[1]):
            test = np.transpose(temp)
            items = deque(test[j])
            items.rotate(Y_shift)
            l2.append(items)
        
        temp2 = np.array(l2, dtype = np.float32)
        l3.append(np.transpose(temp2))
    
    items = deque(l3)
    items.rotate(X_shift) # Swapped to X from previous.
    return np.array(items, dtype = np.float32)

def crop_image(image, crop = [(150,450),(135,435),(212,512)]):
    
    for ii in range(len(crop)):
        # print(f'Cropping {ii} axis.')
        image = image.take(indices = range(*crop[ii]), axis = ii)
        # The * is the splat operator, unravels the tuple.
        
    return image 
    

def scale_image(image, scale_type = 'min_max'):
    
    if scale_type == 'min_max':
        img_min = image.min()
        img_max = image.max()
        
        scaled_img = 255 * (image - img_min) / (img_max - img_min)
        
    return scaled_img

def window_image(image, win_min = -400, win_max = 800):
    low_mask = image < win_min
    high_mask = image > win_max
    
    image[low_mask] = win_min
    image[high_mask] = win_max
    
    return image    

###############################################################################
#################### Kailyn's DICOM & DATA PROCESSING ############################
###############################################################################

#def load_dcm(n,data_dir='AnonymizedDICOM'):
def load_dcm(pt, strctSet, plan_name, data_dir):
    """Reads and loads a set of patient data. Includes RTSTRUCT, 
    RTPLAN, and RTDOSE DICOM files. Patient data is assumed to 
    have filenames generated by the output from ARIA in format of
    patient name + structure set/plan name. 
    
    Adjusted by Owen Paetkau May 21st, 2021.

    Parameters
    ----------
    pt : string
        Patient ID, required for identifying DICOM files.
    strctSet : string
        Structure set ID, required for identifying DICOM files.
    plan_name : string
        Plan name, required for identifying imported files.
    data_dir : string, optional
        Folder containing patient data.

    Returns
    -------
    struct : RTSTRUCT type
        Patient RTSTRUCT DICOM object.
    dose : List of RTDOSE type
        Patient RTDOSE DICOM object.
    plan : RTPLAN type
        Patient RTPLAN DICOM object.
    """
    
    #READ IN FILES
    #Fetch structure DICOM and read.
    #_fname = 'RTSTRUCT_'+str(n).zfill(4)+'.dcm'
    _fname = 'RS.' + pt + '.' + strctSet + '.dcm'
    print(_fname)
    struct = pydicom.dcmread(os.path.join(data_dir,_fname))
    
    #Fetch plan DICOM and read.
    #_fname = 'RTPLAN_'+str(n).zfill(4)+'.dcm'
    _fname = 'RP.' + pt + '.' + plan_name + '.dcm'
    print(_fname)
    plan = pydicom.dcmread(os.path.join(data_dir,_fname))
    
    #Fetch dose DICOM and read. 
    #_fname = 'RTDOSE_'+str(n).zfill(4)+'.dcm'
    _fname = glob.glob(data_dir + 'RD.' + pt + '.' + plan_name + '*.dcm')
    dose = []
    
    for ii in np.arange(0,len(_fname)):
        print(_fname[ii])
        dose.append(pydicom.dcmread(os.path.join(_fname[ii])))
    #_fname = 'RD.' + pt + '.' + plan_name + '.CCW.dcm'
    #print(_fname)
    #doseCCW = pydicom.dcmread(os.path.join(data_dir,_fname))
    
    #dose = [doseCW,doseCCW]

    return struct,dose,plan

def batch_anonymize(patdir,save_dir='AnonymizedDICOM'):
    """Anonymizes all DICOM files within a folder.
    
    All subdirectories within the folder are anonymized.
    Each set of patient data (RTSTRUCT, RTDOSE, RTPLAN)
    must be in a separate folder for proper indexing.

    Parameters
    ----------
    folder : string
        The target directory containing patient data.
    save_dir : string, optional
        The save directory for anonymized files, by
        default 'AnonymizedDICOM' in current directory.

    Returns
    -------
    None
    """
#     patdir = os.path.join(os.getcwd(),folder)
    total_files = 0
    print('-'*79)
    print('Starting file anonymizer...\n')
    
    try:
        latest_file = sorted(os.listdir(save_dir))[-1]
        n = int(latest_file[-8:-4])
    except:
        n = 0
        
    if not os.path.exists(save_dir):
        print('Specified save location not found. '
              'Creating new folder',save_dir,'in current directory.\n')
        os.makedirs(save_dir)
        
    for root, dirs, files in os.walk(patdir,topdown=False):
        fname = []
        for name in files:
            if name.endswith('.dcm'):
                fname.append(name)
        if fname:
            n += 1
            print(root)
            _anonymize(root,fname,n,save_dir)
            print('Anonymized.\n')
            total_files += len(fname)
            
    if total_files > 0:
        # try:
        #     os.rename(patdir,os.path.join(os.getcwd(),'PatientData (Processed)'))
        # except:
        #     pass
        print(total_files,'DICOM files successfully anonymized.\n'
              'Anonymized files saved to',save_dir,'in current directory.\n')
    else:
        print('No files were anonymized in',patdir)
    print('\nDone.')
    print('-'*79)

    
###############################################################################
############################ DOSE GRID FUNCTIONS ##############################
###############################################################################

def dose_grid_shape(dose):
    """Get the x, y, z dimensions of the dose grid
    
    Parameters
    ----------
    dose : RT DOSE DICOM
        RT DOSE DICOM file imported using load_dcm function

    Returns
    -------
    x, y, and z dimensions of the dose grid
        
    """
    
    return (
            dose.Columns,
            dose.Rows,
            len(dose.GridFrameOffsetVector),
        )  

def dose_grid_axes(dose):
    """Get the x, y, z axes of the dose grid (in mm)
    
    Parameters
    ----------
    dose : RT DOSE DICOM
        RT DOSE DICOM file imported using load_dcm function

    Returns
    -------
    x, y, z axes of the dose grid (in mm)
        
    """

    return [dose.x_axis, dose.y_axis, dose.z_axis]
     
def scale(dose):
    """Get the dose grid resolution (xyz)
    
    Parameters
    ----------
    dose : RT DOSE DICOM
        RT DOSE DICOM file imported using load_dcm function

    Returns
    -------
    x, y, z dose grid resolution
        
    """
        
    diffs = np.diff(dose.GridFrameOffsetVector)
    if not np.all(np.isclose(diffs, [diffs[0]]*len(diffs))):
        raise NotImplementedError(
                "Non-uniform GridFrameOffsetVector detected. Interpolated "
                "summation of non-uniform dose-grid scales is not supported."
                )
    return np.array(
        [
        dose.PixelSpacing[0],
        dose.PixelSpacing[1],
        dose.GridFrameOffsetVector[1]
        - dose.GridFrameOffsetVector[0],
         ]
    )

def offset(dose):
    """Get the coordinates of the dose grid origin (mm)
    
    Parameters
    ----------
    dose : RT DOSE DICOM
        RT DOSE DICOM file imported using load_dcm function

    Returns
    -------
    coordinates of the dose grid origin (mm)
        
    """

    return np.array(dose.ImagePositionPatient, dtype="float")
        
def dose_grid_coincidence(dose_list):
    """Check dose grid spatial coincidence.

    Parameters
    ----------
    dose_list : list
        List of RTDOSE Files
        
    Returns
    -------
    Boolean True/False coincidence of dose grid objects
    
    """
    
    #For each dose file
    for item in dose_list:
        #Check if they have spatial coincidence
        if (dose_list[0].PixelSpacing == item.PixelSpacing and
        dose_list[0].ImagePositionPatient == item.ImagePositionPatient and
        dose_list[0].pixel_array.shape == item.pixel_array.shape and
        dose_list[0].GridFrameOffsetVector == item.GridFrameOffsetVector):
            continue
        #If they don't, exit
        else:
            return False
    return True

def dose_grid_parameters(dose_list):
    """Check if dose grid parameters are the same

    Parameters
    ----------
    dose_list : list
        List of RTDOSE Files
    Returns
    -------
    Boolean True/False equality of dose grid parameters
    """
    
    #For each dose file
    for item in dose_list:
        if (dose_list[0].DoseSummationType == item.DoseSummationType and
        dose_list[0].DoseType == item.DoseType and
        dose_list[0].DoseUnits == item.DoseUnits and 
        dose_list[0].ImageOrientationPatient == item.ImageOrientationPatient):
                continue
        else:
            return False
    return True

def extract_dose_grid(dose):
    """ Extracts dose grid from RTDOSE object
    
    Get the coordinates of the dose grid origin (mm)
    
    Parameters
    ----------
    dose : RT DOSE DICOM
        RT DOSE DICOM file imported using load_dcm function


    Returns
    -------
    dose_grid: array
        Dose grid object
    """
    dose_grid = dose.pixel_array * dose.DoseGridScaling
    
    return dose_grid
 
def add_arcs(dose_list):
    """ Adds dose grids together, specified from a list
    of Dose DICOM objects.

    Parameters
    ----------
    dose_list : list
        List of RTDOSE files to add together


    Returns
    -------
    combined_grid: array
        Combined dose grid object
        """

    #If the dose grid is coincident
    if dose_grid_coincidence(dose_list) and dose_grid_parameters(dose_list):
    
        #Get grid starting point
        combined_grid = extract_dose_grid(dose_list[0])
        
        #Add each new dose grid on
        for item in dose_list[1:]:
            new_grid = extract_dose_grid(item)
            combined_grid += new_grid
    
    else:
        #Perform an interpolated sum  
        print("Arcs are not coincident/do not have same parameters")
 
    return combined_grid

###############################################################################
#################### COMPUTATIONAL & GEOMETRY FUNCTIONS #######################
###############################################################################

def grid_points(dose_list): 
    """Generates an array of 2D grid points, gives the dimensions of each
    slice in 3D RTDOSE DICOM in patient coordinates.

    Parameters
    ----------
    dose : RTDOSE type
        Patient RTDOSE DICOM object.

    Returns
    -------
    points : array_like
        2D array of coordinate pairs giving [X,Y] position of each pixel in patient coordinates.
    """
    

    #Find the origin of the image in patient coordinates
    X0,Y0,Z0 = dose_list[0].ImagePositionPatient
    
    #Find the number of rows in the dose grid
    rows = dose_list[0].Rows
    #Find the number of columns in the dose grid
    cols = dose_list[0].Columns
    
    #Find the image resolution, given as [space between rows, space between columns]
    xres,yres = dose_list[0].PixelSpacing
    
    #Arange(start,stop,step) returns an array of evenly spaced values
    #Gives location of each pixel in patient coordinates
    X = np.arange(0,cols*yres,yres) + X0 #Want yres, gives space between columns
    Y = np.arange(0,rows*xres,xres) + Y0 #Want xres, gives space between rows
    
    #Return coordinate matrices from coordinate vectors
    xx,yy = np.meshgrid(X,Y)
    
    #Gives you [X,Y] coordinates of each pixel in patient coordinates (mm)
    points = np.column_stack([np.ndarray.flatten(xx),np.ndarray.flatten(yy)])
    
    #Return array of coordinates
    return points

def max_boundary_value(arr):
    """Get the greatest value on the boundary of a 3D numpy array

    Parameters
    ----------
    arr : numpy.array
        Any 3-dimensional array-like object

    Returns
    -------
    float
        Maximum value along any side of the input array
    """
    return np.max(
        [
            np.max([np.max(arr[i, :, :]) for i in [0, -1]]),
            np.max([np.max(arr[:, j, :]) for j in [0, -1]]),
            np.max([np.max(arr[:, :, k]) for k in [0, -1]]),
        ]
    )

def centroid(arr):
    """Find centroid of a blob

    Parameters
    ----------
    arr : numpy.array
        Any 3-dimensional array-like object representing the [X,Y,Z] voxels 
        making up a blob

    Returns
    -------
    x,y,z location of centroid
    """
    length = arr.shape[0]
    sum_x = np.sum(arr[:,0])
    sum_y = np.sum(arr[:,1])
    sum_z = np.sum(arr[:,2])
    
    return sum_x/length, sum_y/length, sum_z/length

def argfind_nearest(array,value):
    """Returns the index of the nearest value
    to a target in an array.

    Parameters
    ----------
    array : array_like
        Input data.
    value : float
        Target value.

    Returns
    -------
    index : int
        Index nearest to the target value.
    """
    array = np.asarray(array)
    index = np.argmin(np.abs(array - value))
    
    return index
    
###############################################################################
########################### STRUCTURE FUNCTIONS ###############################
###############################################################################

def read_structure(struct, dose_list, plan, targets, oars):
    """Organizes patient data into Python dicts.
    
    Includes the name, color and contour outlines of
    each contoured structure. Additionally provides the
    3D coordinates of each voxel, respective dose within
    the contoured structure, and total volume of the
    structure (for all structures excluding 'BODY').
    Various DVH metrics are also included, specifically
    D98, D90, D50, V100, V150, V200. Vxx are only 
    computed for clinical target volumes.

    Parameters
    ----------
    struct : RTSTRUCT type
        Patient RTSTRUCT DICOM object.
    plan : RTPLAN type
        Patient RTPLAN DICOM object.
    dose_list : list
        Patient RTDOSE DICOM object.
    targets: list 
        List of target volumes (in string 
        format) to be extracted
    oars: list 
        List of organs-at-risk (in string
        format) to be extracted

    Returns
    -------
    structures : dict
        Dict of metrics for each structure in struct.
    """
    structures = {}
    
    approved_structures = targets + oars
    #Print all structures
    print("Contoured Structures:")
    for i in range(len(struct.StructureSetROISequence)):
        print(struct.StructureSetROISequence[i].ROIName)
        
    print("\n \n")
    flag = False
   # if 'CTVN_IMC_R' not in [struct.StructureSetROISequence[i].ROIName.upper().replace(' ','') for i in range(len(struct.StructureSetROISequence))]:
    #    flag = True

    #For each contour sequence in the structure DICOM
    for i,contour in enumerate(struct.ROIContourSequence): 
        organ = {}
        organ['name'] = struct.StructureSetROISequence[i].ROIName
        organ['name'] = organ['name'].upper().replace(' ','')

        
        if organ['name'] in approved_structures:
            organ['color'] = np.array(contour.ROIDisplayColor,dtype=float)/255
            organ['contours'] = list(map(_reshape_data,contour.ContourSequence))
            
            if not organ['name'] == 'MATCHPOINTS' and not organ['name'] == 'BODY':
                #Get voxels that belong to organ
                organ['voxels'] = organ_voxels(organ,grid_points(dose_list))
                #Get dose grid
                organ['dose'] = total_rad_calc(dose_list, organ['voxels'])
                #Get base dose metrics
                organ['mean dose'] = np.average(organ['dose'])
                organ['minimum dose'] = np.min(organ['dose'])
                organ['maximum dose'] = np.max(organ['dose'])
                
                organ['volume (cc)'] = organ_volume(organ)
         
                organ['DVH'] = DVH(organ)
                
                #If the organ is a target:
                if organ['name'] in targets:
                    #Calculate D98
                    organ['D98'] = Dxx(organ,98)
                    #Calculate D90
                    organ['D90'] = Dxx(organ,90)
                    #Calculate D50
                    organ['D50'] = Dxx(organ,50)

                    #Calculate V1
                    # organ['V1'] = Vxx(organ,plan,1)
                    # #Calculate V20
                    # organ['V20'] = Vxx(organ,plan,20)
                    # #Calculate V100
                    # organ['V100'] = Vxx(organ,plan,100)
                    # #Calculate V150
                    # organ['V150'] = Vxx(organ,plan,150)
                    # #Calculate V200
                    # organ['V200'] = Vxx(organ,plan,200)
                    
                #If the organ is an organ-at-risk    
                if organ['name'] in oars:
                    organ['D2cc'] = Dxx_cc(organ,2)
                    organ['D2cc EQD2'] = EQD2_10(organ['D2cc'])
                    organ['D0.1cc'] = Dxx_cc(organ,0.1)
                    organ['D0.1cc EQD2'] = EQD2_10(organ['D0.1cc'])
                structures[organ['name']] = organ   

        elif flag:
            
            print("Organ name ", organ['name'], "not in list of target or organ-at-risk structures.")
            choice = input("Rename to valid structure? (Type -> RENAME)")
            
            if choice.upper() == "RENAME":
                    flag = False
                    organ['name'] = input("Type structure name. Options: ")
                    organ['name'] = organ['name'].upper()
                    organ['color'] = np.array(contour.ROIDisplayColor,dtype=float)/255 
                    organ['contours'] = list(map(_reshape_data,contour.ContourSequence))
                    
                    if not organ['name'] == 'BODY' and not organ['name'] == 'MATCHPOINTS':
                        #Get voxels that belong to organ
                        organ['voxels'] = organ_voxels(organ,grid_points(dose_list))
                        #Get dose grid
                        organ['dose'] = total_rad_calc(dose_list, organ['voxels'])
                        #Get base dose metrics
                        organ['mean dose'] = np.average(organ['dose'])
                        organ['minimum dose'] = np.min(organ['dose'])
                        organ['maximum dose'] = np.max(organ['dose'])
                        
                        organ['volume (cc)'] = organ_volume(organ)
                 
                        organ['DVH'] = DVH(organ)
                        
                    if organ['name'] in targets:
                        #Calculate D98
                        organ['D98'] = Dxx(organ,98)
                        #Calculate D90
                        organ['D90'] = Dxx(organ,90)
                        #Calculate D50
                        organ['D50'] = Dxx(organ,50)

                        #Calculate V20
                        # organ['V20'] = Vxx(organ,plan,20)
                        # #Calculate V100
                        # organ['V100'] = Vxx(organ,plan,100)
                        # #Calculate V150
                        # organ['V150'] = Vxx(organ,plan,150)
                        # #Calculate V200
                        # organ['V200'] = Vxx(organ,plan,200)
                    
                    #If the organ is an organ-at-risk    
                    if organ['name'] in oars:
                        organ['D2cc'] = Dxx_cc(organ,2)
                        organ['D2cc EQD2'] = EQD2_10(organ['D2cc'])
                        organ['D0.1cc'] = Dxx_cc(organ,0.1)
                        organ['D0.1cc EQD2'] = EQD2_10(organ['D0.1cc'])
                    structures[organ['name']] = organ  
                    
            else:
                    continue  
   
               
    return structures    

def organ_voxels(organ,points):
    """Determines the coordinates of all voxels within
    the target structure contour.
    
    Contour lines from RTSTRUCT provide outlines describing
    a surface. This function translates that surface into
    the 3D volume enclosed by it.

    Parameters
    ----------
    organ : dict
        Structure dict object from read_structure().
    points : array_like
        2D grid coordinates from grid_points()

    Returns
    -------
    voxels : array_like
        Array of N voxel coordinates in 3D (x,y,z), dim Nx3.
    """
    
    #Generate temporary dimension for np.vstack()
    voxels = np.array(['x','y','z']) 
    
    #For each slice in a given organ contour
    for axialslice in organ['contours']:
        
        polygon = np.column_stack([axialslice[0],axialslice[1]])
        path = mpl.path.Path(polygon)
        
        #Generate a mask of points
        mask = path.contains_points(points)
        z_coord = np.ones(np.sum(mask))*axialslice[2][0]
        xyz_coord = np.column_stack([points[mask],z_coord])
        voxels = np.vstack([voxels,xyz_coord])
        
    voxels = np.delete(voxels,0,0) #removing temporary dimension ['x','y','z']
    voxels = np.array(voxels,dtype=float).reshape([-1,3])
    
    return voxels

def organ_volume(organ):
    """Computes the volume of a target structure.

    Parameters
    ----------
    organ : dict
        Structure dict object from read_structure().

    Returns
    -------
    volume : array_like
        Volume of the structure in cc.
    """
    diff = np.abs(np.diff(organ['voxels'],axis=0))
    dx = np.min(diff[:,0][np.nonzero(diff[:,0])])
    dy = np.min(diff[:,1][np.nonzero(diff[:,1])])
    dz = np.min(diff[:,2][np.nonzero(diff[:,2])])
    volume = organ['voxels'].shape[0] * dx * dy * dz / 1000
    return volume

def closest_OAR_voxels(OAR_name,target_name, structures):
    """ Identifies the voxels composing the nearest 1.5cc
    of an OAR to a specified target volume (using 1000
    voxels at 1.5mm^3 per voxel).

    Parameters
    ----------
    structures : dict
        Structures dict output by read_structure().
    OAR_name : string
        Specified organ-at-risk 
    target_name : string 
        Specified target volume

    Returns
    -------
    voxels : ???
        Voxels composing closest 1.5 cc to the target volume
    """
    
    target = structures[target_name]
    OAR = structures[OAR_name]
    
    target_centroid = np.mean(target['voxels'], axis = 0)
    centre_dist = spdist.cdist([target_centroid],OAR['voxels'])[0]
    order = np.argsort(centre_dist)
    voxels = OAR['voxels'][order][:1000]
    
    nearest_voxels = spdist.cdist(target['voxels'],voxels,'euclidean').min(axis=0)

    return nearest_voxels

def closest_OAR_proximity(OAR_name,target_name, structures):
    """ Computes the mean distance from the nearest
    1.5cc of an OAR to the HRCTV (using 1000 voxels
    at 1.5mm^3 per voxel).

    Parameters
    ----------
    structures : dict
        Structures dict output by read_structure().
    OAR_name : string
        Specified organ-at-risk 
    target_name : string 
        Specified target volume

    Returns
    -------
    proximity : dict
        Dict of OAR proximities to the HRCTV in mm.
    """
    
    nearest_voxels = closest_OAR_voxels(OAR_name, target_name, structures)
    
    proximity = nearest_voxels.mean()
    
    return proximity

###############################################################################
############################## PLAN FUNCTIONS #################################
###############################################################################

def get_prescription(plan):

    #If brachytherapy
    if plan.FractionGroupSequence[0].NumberOfBeams == 0:
        #Does this give per fraction or total?
        prescription = plan.FractionGroupSequence[0].ReferencedBrachyApplicationSetupSequence[0].BrachyApplicationSetupDose
        
    #If EBRT
    else:
        #This gives total prescription dose (not per fraction) in Gy
        prescription = plan.DoseReferenceSequence[0].DeliveryMaximumDose
    
    return prescription
       
###############################################################################
############################## DOSE FUNCTIONS #################################
###############################################################################

def total_rad_calc(dose_list,voxels):
    """Computes the total 3D dose distribution of 
    all control points in the fraction.
    
    Voxels within the bounds of the 3D dose
    distribution found in patient RTDOSE are
    calculated using linear interpolation.

    Parameters
    ----------
    dose_list : list
        List of dose DICOMs to add together
    voxels : array_like
        One or more 3D points.

    Returns
    -------
    dose_Gy : array_like
        Total dose (Gy) delivered to each respective
        3D point in voxels.
    """
    voxels_copy = voxels.copy()
    if voxels_copy.shape == (3,):
        voxels_copy = np.array([voxels_copy])
    voxels_copy[:,[0, 2]] = voxels_copy[:,[2, 0]]
    
    #FIND LOCATION OF EACH PIXEL
    #Find the origin of the image in patient coordinates
    X0,Y0,Z0 = dose_list[0].ImagePositionPatient
    #Find the number of rows in the image
    rows = dose_list[0].Rows
    #Find the number of columns in the image
    cols = dose_list[0].Columns
    #Find the image resolution, given as [space between rows, space between columns]
    xres,yres = dose_list[0].PixelSpacing
    
    #Arange(start,stop,step) returns an array of evenly spaced values
    #Gives location of each pixel in patient coordinates
    #Want yres, gives space between columns
    X = np.arange(0,cols*yres,yres) + X0 
    #Want xres, gives space between rows
    Y = np.arange(0,rows*xres,xres) + Y0 
    
    
    #GridFrameOffsetVector = array containing dose image plane offsets (mm) of the dose image frames in a multi-frame dose. Required if multi-frame pixel data are present and Frame Increment Pointer (0028,0009) points to Grid Frame Offset Vector (3004,000C). 
    Z = np.asarray(dose_list[0].GridFrameOffsetVector) + Z0

    combined_grid = add_arcs(dose_list)
    
    #DOSE CALCULATION FOR SLICE
    #Dose grid was interpolated when extracted
    dose_interp = interp.RegularGridInterpolator((Z, Y, X), combined_grid)
    dose_Gy = dose_interp(voxels_copy)
 
    return dose_Gy

def DVH(organ,maxdose=70.0,res=999):
    """Computes the dose-volume histogram for a structure.

    Parameters
    ----------
    organ : dict
        Structure dict object from read_structure().
    maxdose : float, optional
        Upper limit for computing the DVH, by default 70.0 Gy.
    res : int, optional
        Resolution of the DVH (number of points), by default 999.

    Returns
    -------
    doserange : array_like
        Range of doses in Gy (x-axis of DVH).
    proportion : array_like
        Percent volume recieving respective dose in doserange.
    """
    doserange = np.linspace(0,maxdose,res)
    proportion = np.array([(organ['dose'] > dose).sum() for dose in doserange]
                          ,dtype=float)
    proportion *= 100.0 / len(organ['dose'])
    
    return doserange,proportion
    
def Dxx(organ,value):
    """Computes the minimum dose delivered to a percentage
    volume of a target structure (i.e. D98, D90, etc.).

    Parameters
    ----------
    organ : dict
        Structure dict object from read_structure().
    value : float
        Percentage volume of structure.

    Returns
    -------
    Dxx : float
        Minimum dose in structure volume (in Gy).
    """
    doserange,proportion = organ['DVH']
    Dxx = doserange[argfind_nearest(proportion,value)]

    return Dxx

def Dxx_cc(organ,vol):
    """Computes the minimum dose delivered to a volume
    of the target structure (i.e. D2cc, D0.1cc, etc.).

    Parameters
    ----------
    organ : dict
        Structure dict object from read_structure().
    vol : float
        Volume of interest in cc.

    Returns
    -------
    Dxx_cc : float
        Minimum dose in structure volume (in Gy).
    """
    
    value = vol / organ['volume (cc)'] * 100
    Dxx_cc = Dxx(organ,value)

    return Dxx_cc

def Vxx(organ,plan,value):
    """Computes the volume recieving a percentage
    of the prescription dose (i.e. V100, V150, etc.).

    Parameters
    ----------
    organ : dict
        Structure dict object from read_structure().
    plan : RTPLAN type
        Patient RTPLAN DICOM object.
    value : float
        Percentage of the prescription dose.

    Returns
    -------
    Vxx : float
        Volume as a percentage of structure volume.
    """
    
   
    prescription = get_prescription(plan)
    doserange,proportion = organ['DVH']
    target = prescription * value/100
    Vxx = proportion[argfind_nearest(doserange,target)]# * organ['volume (cc)']/100

    return Vxx

def coverage_index(organ, plan):

    total_volume = organ['volume (cc)']
    volume_ref_dose = Vxx(organ, plan, 100) * organ['volume (cc)'] / 100

    CI = volume_ref_dose/total_volume
    
    return CI

def external_volume_index(target_organ, oar, plan):
    
    normal_volume_ref_dose = Vxx(oar, plan, 100) * oar['volume (cc)'] / 100
    total_target_volume = target_organ['volume (cc)']
    
    EI = normal_volume_ref_dose/total_target_volume
    
    return EI

def dose_homogeneity_index(organ,plan):

    volume_ref_dose = Vxx(organ, plan, 100) * organ['volume (cc)'] / 100
    volume_15_ref_dose = Vxx(organ, plan, 150) * organ['volume (cc)'] / 100

    DHI = (volume_ref_dose -  volume_15_ref_dose)/volume_ref_dose
    
    return DHI

def overdose_volume_index(organ, plan):
    
    volume_ref_dose = Vxx(organ, plan, 100) * organ['volume (cc)'] / 100
    volume_over_dose = Vxx(organ, plan, 200) * organ['volume (cc)'] / 100

    ODI  = volume_over_dose/volume_ref_dose
    
    return ODI

def dose_nonuniformity_ratio(organ,plan):

    volume_ref_dose = Vxx(organ, plan, 100) * organ['volume (cc)'] / 100
    volume_15_ref_dose = Vxx(organ, plan, 150) * organ['volume (cc)'] / 100
    
    DNR = volume_15_ref_dose/volume_ref_dose
    
    return DNR 
    
def EQD2_3(dose):
    """Returns the equivalent dose in 2Gy fractions
    """
    return dose * (1+dose/3)*0.6

def EQD2_10(dose):
    """Returns the equivalent dose in 2Gy fractions
    """
    return dose * (1+dose/10)/1.2


###############################################################################
############################ PLOTTING FUNCTIONS ###############################
###############################################################################

def axisEqual3D(ax):
    """Sets the 3D axes to aspect ratios of equal
    unit length. Written by 'Ben' on stackoverflow.
    
    https://stackoverflow.com/questions/8130823
    /set-matplotlib-3d-plot-aspect-ratio
    
    Parameters
    ----------
    ax : Matplotlib axes object
        Current 3D plotting axis.

    Returns
    -------
    None
    """
    extents = np.array([getattr(ax, 'get_{}lim'.format(dim))() for dim in 'xyz'])
    sz = extents[:,1] - extents[:,0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize/2
    for ctr, dim in zip(centers, 'xyz'):
        getattr(ax, 'set_{}lim'.format(dim))(ctr - r, ctr + r)
        
def plot_HRCTV(n):
    """Generates a 3D render of the HRCTV contours and plan control points.
    
    Parameters
    ----------
    n : int
        Fraction index.
    """
    
    #Load desired fraction
    fraction = load_fraction(n)
    #Pull control points from dictionary containing applicator information
    points = fraction['APPLICATOR']['POINTS']
    
    
    #Initialize Plot
    pink = np.array([1,0,1]) #Set colour to pink
    fig = plt.figure(figsize=(6,6),num='Fraction '+str(n))
    ax = fig.add_subplot(111, projection='3d')
    
    
    for key,value in points.items():
        #Check to see that there is an entry in value
        if value:

            if key == 'ring':
                ax.scatter(*np.array(value).T,c='black',depthshade=False);
            else:
                ax.plot(*np.array(value).T,'ko-',lw=2);

    for axialslice in fraction['STRUCTURES']['HRCTV']['contours']:
        X,Y,Z = axialslice
        ax.plot(X,Y,Z, color=pink,alpha=0.20,lw=10)

    patchcolor = mpl.patches.Patch(facecolor=pink,label='HRCTV')
    legend = ax.legend(handles=[patchcolor])
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    axisEqual3D(ax)
    plt.show()
    
def plot_structures(structures,show=None,hide=[],n=0):
    """Generates a 3D scatterplot of structure contours.
    """
    fig = plt.figure(figsize=plt.figaspect(1.0),num='Fraction '+str(n))
    ax = fig.add_subplot(111, projection='3d')
    
    legend = []

    if show is None:
        show = structures.keys()
    hide.append('BODY')
    for organ in structures.values():
        try:
            if not organ['name'] in hide and organ['name'] in show:
                legend.append(mpl.patches.Patch(facecolor=organ['color'],
                                                label=organ['name']))
                for axialslice in organ['contours']:
                    X,Y,Z = axialslice
                    ax.scatter(X, Y, Z,c=[organ['color']],depthshade=False,
                               alpha=0.20,edgecolors='none');
                    ax.plot(X,Y,Z, color=organ['color'],alpha=0.20)
        except KeyError:
            pass

    legend = ax.legend(handles=legend)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    axisEqual3D(ax)
    
    plt.show()

def plot_dose(dose,z_slice): 
    """Generates a dose plot for one axial slice.
 
    Parameters
    ----------
    dose : RTDOSE type
        Patient RTDOSE DICOM object.
    
    z_slice: int
        Axial slice number
    """
    #FIND LOCATION OF EACH PIXEL
    #Find the origin of the image in patient coordinates
    X0,Y0,Z0 = dose.ImagePositionPatient
    #Find location of each slice in patient coordinates
    Z = Z0 + z_slice
    #Find the number of rows in the image
    rows = dose.Rows
    #Find the number of columns in the image
    cols = dose.Columns
    #Find the image resolution, given as [space between rows, space between columns]
    xres,yres = dose.PixelSpacing
    
    #DOSE CALCULATION FOR SLICE
    #Find dose for given slice
    #DoseGridScaling gives factor to multiply pixel array by in order to get dose at each pixel (per slice)
    #dose_Gy[z_slice]
    dose_Gy = dose.pixel_array * dose.DoseGridScaling
    
    #GENERATE PLOT
    plt.close() 
    #Plot z-plane dose
    #Extent = bounding box that the image will fill (upper,lower)
    img = plt.imshow(dose_Gy[z_slice], 
                     #Go from X0 origin of slice to the number of columns*resolution 
                     extent=[X0,X0+(cols-1)*yres,Y0,Y0+(rows-1)*xres],cmap = "inferno")
    #Add colorbar to plot
    cbar = plt.colorbar()
    #Label colorbar
    cbar.set_label('Dose (Gy)')
    #Plot axes
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('3D Dose at Z = '+str(Z)+'mm')
    plt.show()

def plot_DVH(structures,n,state=None):
    """Generates a DVH plot for multiple structures, excluding the body structure.
    
    Parameters
    ----------
    structures : dict
        Structures dict output by read_structure().
    n : int
        Fraction index.

    Returns
    -------
    fig : matplot figure
        Figure showing the DVH plot for your fraction for all structures.
    """
    
    
    if state is None:
        state = np.ones(len(structures))
    
    #Initialize figure size
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)
    
    #For each organ in the structure dictionary (given with structures.values())
    for i,organ in enumerate(structures.values()):
        if state[i]: #???
            doserange,proportion = organ['DVH']
            ax.plot(doserange,proportion,c=organ['color'],label=organ['name'])
    ax.legend()
    ax.set_xlabel('Dose [Gy]')
    ax.set_ylabel('% Volume')
    ax.set_ylim(0,101)
    try:
        ax.set_xlim(0,max(doserange))
    except:
        pass
    ax.grid(1)
 
    return fig
   
###############################################################################
############################ INTERNAL FUNCTIONS ###############################
###############################################################################

def _validate_attr_equality(obj_1, obj_2, attr):
    """Assess the equality of the provided attr between two objects.
    Send warning if unequal.

    Parameters
    ----------
    obj_1 : object
        Any object with an `attr` attribute that is comparable by !=
    obj_2 : object
        Any object with an `attr` attribute that is comparable by !=
    attr : str
        The attribute to be compared between obj_1 and obj_2
    """
    val_1 = getattr(obj_1, attr)
    val_2 = getattr(obj_2, attr)
    if val_1 != val_2:
        warn("Different %s values detected:\n%s\n%s" % (attr, val_1, val_2))
        return False
    return True
    
def _person_names_callback(dataset, data_element):
    if data_element.VR == 'PN':
        data_element.value = 'anonymous'

def _curves_callback(dataset, data_element):
    if data_element.tag.group & 0xFF00 == 0x5000:
        del dataset[data_element.tag]

def _anonymize(path,fname,n,save_dir):     
    for i,name in enumerate(fname):
        dataset = pydicom.dcmread(os.path.join(path,name))
        dataset.PatientID = 'id'
        dataset.walk(_person_names_callback)
        dataset.walk(_curves_callback)

        if 'OtherPatientIDs' in dataset:
            delattr(dataset, 'OtherPatientIDs')
        if 'OtherPatientIDsSequence' in dataset:
            del dataset.OtherPatientIDsSequence
        tag = 'PatientBirthDate'
        if tag in dataset:
            dataset.data_element(tag).value = '00000000'   
        tag = 'PatientSex'
        if tag in dataset:
            dataset.data_element(tag).value = '0'
        dataset.remove_private_tags()
        
        save_name = dataset.Modality+'_'+str(n).zfill(4)+'.dcm'
        print(save_name)
        dataset.save_as(os.path.join(save_dir,save_name))
 
def _metrics_cmap(val):
    #colourmap for metrics
    if val == 'PASS':
        color = 'lime'
    elif val == 'CAUTION':
        color = 'yellow'
    elif val == 'FAIL':
        color = 'red'
    return 'background-color: %s' % color

def _key_walk(dict_,entry):
    if type(entry) is list:
        for item in entry:
            if type(item) is list:
                key = item[0]
                if len(item) > 1:
                    _key_walk(dict_[key],item[1:])
                else:
                    _key_walk(dict_[key],item)
            else:
                if type(dict_[item]) == dict:
                    print(item)
                    viewdict(dict_[item])
                else:
                    print(dict_[item])
    else:
        if type(dict_[entry]) == dict:
            print(entry)
            viewdict(dict_[entry])
        else:
            print(dict_[entry])

def _reshape_data(loop):
   data = loop.ContourData
   return np.reshape(np.array(data),(3, len(data) // 3),order='F')