The dataset is initially in the 3D format which is .nii files. It is stored as 3 folders – AD, CN, MCI. The .nii extension stores the metadata such as the image array, affine data, dimension, etc.

The following are the steps involved in pre-processing :
- Skull Stripping
- Bias Correction
- Segmentation
- Extracting 2D images from 3D segmented MRI image

The above mentioned preprocessing steps are pipelined with the help of  a shell script

**1. Skull Stripping**

>The process of removing skull from the 3D image of brain as skull is considered to be a noise.

- Done using deepbrain library of dipy where deepbrain has a well maintained class Extractor .

- Initially an instance of Extractor is created and it’s function run() is called which returns the set of probabilities of a pixel being part of a skull or not.

- Using the probability values a mask is generated which contains only the brain region.

- Thus the pixel values which does not include in the mask region is made to 0, so that we are left only with the brain.

- The brain image is then stored back as a 3D file using the function Nifti1Image  of nibabel library which has the affine data as one of it’s parameter.

**2. Bias Correction**
>The bias field correction algorithm is a method for correcting low frequency intensity non-uniformly present in the MRI image. It is the process of eliminating the intensity variation of same tissues on the basis of location of the tissue within the image.

- Why minimized? To improve the accuracy of computer-aided diagonosis.

- The bias correction is performed using the N4BiasFieldCorrection() function available in SimpleITK library.

- The 3D file is loaded using the ReadImage() function and the pixel values are required to be a real pixel type of Float 32 inorder to perform the correction.This is done using the Cast() function.

- A mask is created from input image using the concept of Thresholding which is the basic form of segmentation. It simply labels the pixels of an image based on the intenisty range.

- Using the mask, the regions are identified where non-uniformity is present.

- Thus the N4BiasFieldCorrection() function takes the mask as well as the original image as input and performs the correction.

- The image is then written back as a 3D file using the WriteImage() function.

**3. Segmentation and 2D Image Extraction**
>It is the process of dividing the regions of the brain into 3 classes – Grey Matter(GM),White Matter(WM),Cerebro Spinal Fluid(CSF).This type of segmentation is known also known as segmentation based on three-tissue probabilty. Once segmented, the 2D image slices are extracted using matplotlib

- The segmentation is performed using the Tissue Classifier HMRF class of dipy library.

- It requires three parameters – the actual data,the number of classes and beta(the smoothness factor of segmentation).

- It returns the final , initial segmentation and PVE.

- Partial VolumEffect -PVE refers to the set of voxels/pixels which has a combination of different types of tissues.

- It is then reconstructed to a 3D file using the initial affine data from which the 2D slices are obtained.

Since the data after pre- processing were 2D images, it helped to reduce the dataset size to 3GB and the dataset was then augmented using Image Data Generator to finally get a total of 1,80,000 grayscale images in png format with dimension of 256x256 . In this process of augmentation the various transformations used were horizontal flip, zoom range of 0.04, height shift range of 0.04, width shift range of 0.04, shear range of 0.04.


