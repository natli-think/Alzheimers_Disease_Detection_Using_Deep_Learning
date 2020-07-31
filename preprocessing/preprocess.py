from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from dipy.segment.tissue import TissueClassifierHMRF
from dipy.io.image import load_nifti_data
from __future__ import print_function
from deepbrain import Extractor
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib

import time
import os
import sys
import subprocess
import shutil
import numpy as np
  
class Preprocess:
  '''

  This is a class to perform preprocessing on an input 3D image.
  
  Parameters of member functions :
       src_path : The source path where the MRI image in .nii extension is present.
       dst_path : The destination path where you would like to save the output of the function. Remember, this will also be the source path for the next preprocess.
    
  '''

  def __init__(self):
    '''
     This is is constructor for Preprocess class  
    '''
    
    print('Start preprocessing....')

  def strip_the_skull(self,src_path,dst_path):
    """
    
    The function is to strip the skull from the input MRI brain image.
    
    Variables & its function:
      img => Loads the data from 3D input file which is then later converted to numpy array using get_fdata()
      affine => Stores the affine data of the input file
      ext => Creates an instance of class extractor
      prob => Set of probabilities are calculated and stored
      mask => The region of brain is identified using threshold 0.5
      brain => Unwanted pixels are made to 0 and is converted back to a nifti image using NiftiImage function. It is then saved as 3D using nibabel library
    
    """
    img = nib.load(src_path)
    affine = img.affine
    img = img.get_fdata()
    ext = Extractor()
    prob = ext.run(img)
    mask = prob > 0.5
    brain = img[:]
    brain[~mask] = 0
    brain = nib.Nifti1Image(brain, affine)
    nib.save(brain, dst_path)

  def __add_pad(self,image, new_height=256, new_width=256):
    """
    
    This private function is used to add padding to the final image, so that all the output images in the preprocessing phase has uniform dimension.
    
    """
    height, width = image.shape
    final_image = np.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    #print(final_image.shape,'<--Shape of final image')
    return final_image

    
  def get_noiseless_image(self,src_path,dst_path):
    """
    
    The function is to perform bias correction and produce uniformity throughout the input 3D image. It is performed using SimpleITK library
    
    Variables & its function:
    inputImage => Read and input image from source path
    img_data => Type Cast the pixels of the image
    img_mask => The regions of non-uniform intensity is identified
    corrected_img => N4 Bias Field correction is run and returned back to destination using WriteImage inbuilt function.
    
    """
    try:
      inputImage = sitk.ReadImage(src_path)
      img_data=sitk.Cast(inputImage,sitk.sitkFloat32) 
      img_mask=sitk.BinaryNot(sitk.BinaryThreshold(img_data, 0, 0))
      corrected_img = sitk.N4BiasFieldCorrection(inputImage, img_mask)
      sitk.WriteImage(corrected_img, dst_path)
      print('Succesfully performed N4 Bias Correction')
    
    except RuntimeError:
      #Runs if any stmts of Bias Correction fails
      print('Failed on :' + src_path)  
  
  def do_segmentation(self,src_path,dst_path):
    """
    
    This function is used to segment the input image into a segmented image based on grey matter, white matter and csf. These are the features used to
    detect the stage of AD 

    Variables & its function:
    nclass => value is initialized as 3 inorder to divide to 3 classes-GM,WM,CSF
    beta => The smoothness factor of segmentation
    t0 => Stores the time before segmentation
    hmrf => Create an instance of class TissueClassifierHMRF
    t1 => Store the time after segmentation
    total_time => Calculate the total time taken for segmentation
    brain => Reconstructing the segmented brain image as a 3D file which is then saved to the destination

    """
    #To get only the image data array
    t1 = load_nifti_data(src_path)
    print('t1.shape (%d, %d, %d)' % t1.shape)
    #To load the entire nifti file
    t2 = nib.load(src_path)
    nclass = 3
    beta = 0.1
    t0 = time.time()
    hmrf = TissueClassifierHMRF()
    #Perform segmentation
    initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
    t1 = time.time()
    total_time = t1-t0
    print('Total time:' + str(total_time))
    print(final_segmentation.shape)
    brain = nib.Nifti1Image(final_segmentation,t2.affine)
    print('Segmentation performed successfully')
    nib.save(brain, os.path.join(dst_path))
  
  def return_2D_image(self,src_path,dst_path):
    """
    
    This function is used to extract 2D image from the segmented 3D image using matplotlib. Here the images are saved with random number names in 
    grayscale format. 
    
    The images are 3D array of shape (182, 218, 182)
    Choose the dimension : 0 (sagittal), 1 (coronnal) or 2 (axial)
    Choose the slice (between 0 and 182 or 218)
    
    """
    image_1 = nib.load(src_path).get_data()
    plt.figure(figsize=(20, 5))
    print('Shape of the MRI : {}'.format(image_1.shape))
    val = [0,1,2]
    for i in val:
      plt.subplot(131)
      plt.axis('off')
      plt.style.use('grayscale')
      final_image = self.__add_pad(np.take(image_1,100,i))
      plt.imshow(final_image)
      img_name = str(np.random.random_integers(350))+'.png'
      plt.savefig(os.path.join(dst_path,img_name),orientation = 'portrait',transparent = True, bbox_inches = 'tight',pad_inches=0)
