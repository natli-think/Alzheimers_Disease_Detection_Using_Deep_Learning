#The preprocessing steps are performed

from nipype.interfaces.ants.segmentation import N4BiasFieldCorrection
from dipy.segment.tissue import TissueClassifierHMRF
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

  def skull_strip(src_path,dst_path):
    img = nib.load(src_path)
    affine = img.affine
    img = img.get_fdata()
    ext = Extractor()
    prob = ext.run(img)
    mask = prob > 0.5
    brain = img[:]
    brain[~mask] = 0
    brain = nib.Nifti1Image(brain, affine)
    nb.save(brain, os.path.join(dst_path, '/skull_stripped.nii'))
    
  def bias_correction(src_path,dst_path):
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
  
  def segmentation(src_path,dst_path):
    #To get only the image data array
    t1 = load_nifti_data(src_path)
    print('t1.shape (%d, %d, %d)' % t1.shape)
    #To load the entire nifti file
    t2 = nib.load(src_path)
    nclass = 3
    beta = 0.1
    t0 = time.time()
    hmrf = TissueClassifierHMRF()
    initial_segmentation, final_segmentation, PVE = hmrf.classify(t1, nclass, beta)
    t1 = time.time()
    total_time = t1-t0
    print('Total time:' + str(total_time))
    print(final_segmentation.shape)
    brain = nib.Nifti1Image(final_segmentation,t2.affine)
    print('Segmentation performed successfully')
    nib.save(brain, os.path.join(dst_path))
  
  def add_pad(image, new_height=256, new_width=256):
    height, width = image.shape
    final_image = np.zeros((new_height, new_width))
    pad_left = int((new_width - width) / 2)
    pad_top = int((new_height - height) / 2)
    
    # Replace the pixels with the image's pixels
    final_image[pad_top:pad_top + height, pad_left:pad_left + width] = image
    return final_image
  
  
  def final_two_dimensional_image(src_path,dst_path):
    image_1 = nib.load(src_path).get_data()
    # The images are 3D array of shape (182, 218, 182)
    # Choose the dimension : 0 (sagittal), 1 (coronnal) or 2 (axial)
    # Choose the slice (between 0 and 182 or 218)
    plt.figure(figsize=(20, 5))
    print('Shape of the MRI : {}'.format(image_1.shape))
    val = [0,1,2]
    for i in val:
      plt.subplot(131)
      plt.axis('off')
      plt.style.use('grayscale')
      final_image = add_pad(np.take(image_1,100,i))
      plt.imshow(final_image)
      img_name = str(np.random.random_integers(350))+'.png'
      plt.savefig(os.path.join(dst_path,img_name),orientation = 'portrait',transparent = True, bbox_inches = 'tight',pad_inches=0)
  
