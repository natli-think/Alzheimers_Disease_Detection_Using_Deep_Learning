# Alzheimers_Disease_Detection_Using_Deep_Learning

Alzheimer’s is a neurological disorder. The upcoming reports suggest, it as the sixth leading cause of death. The main three stages of disease are: Mild Alzheimer’s, Moderate Alzheimer’s and Severe Alzheimer’s. The disease is not a curable one but earlier detection of disease can help prevent worsening of case and slow down the brain tissue damage Normally MRI images  are analysed to identify whether the person has Alzheimer’s  or not. This method can be time consuming and might even lead to misdiagnosis. Hence we deployed Deep Learning as the method deployed for better accuracy in prediction.



# Abstract

This project aims to detect Alzheimer's Disease from the MRI scans ( in .nii extension) given as input. It involves several pre-processing steps such as skull stripping, bias correction and segmentation. Once the segmentation is completed, three 2D slices (axial, coronnal, saggital) are extracted from the segmented MRI Image. In this manner the dataset is prepared which is then used for training a simple CNN. Here, the problem is approached in two methods. One method is to prepare a single dataset wherein the axial,coronnal and saggital images are present within each class which is then fed into a single classifier. Another method is to prepare three datasets (each contains only one kind of data) which is then fed into three classifers (Axial, Coronnal and Saggital Classifier). When compared, it was observed that the combined classifier (**95%**) has more accuracy than the single classifier (**86%**)

## Data

In this project, we have obtained the brain MRI Images from **ADNI** website ([http://adni.loni.usc.edu/](http://adni.loni.usc.edu/)).

ADNI is an ongoing, multicenter cohort study, started from 2004. It focuses on understanding the diagnostic and predictive value of Alzheimers disease specific biomarkers. Our data included the data of 1,500 patients which resulted in a total of 1,80,000 images after augmentation.

### **Image preprocessing**
The images are preprocessed in order to produce the dataset and more detailed information can be found here ([here](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/preprocessing/description.md))
