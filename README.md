# Alzheimers_Disease_Detection_Using_Deep_Learning

Alzheimer’s is a neurological disorder. The upcoming reports suggest, it as the sixth leading cause of death. The main three stages of disease are: Mild Alzheimer’s, Moderate Alzheimer’s and Severe Alzheimer’s. The disease is not a curable one but earlier detection of disease can help prevent worsening of case and slow down the brain tissue damage Normally MRI images  are analysed to identify whether the person has Alzheimer’s  or not. This method can be time consuming and might even lead to misdiagnosis. Hence we deployed Deep Learning as the method deployed for better accuracy in prediction.



# Abstract

This project aims to detect Alzheimer's Disease from the MRI scans ( in .nii extension) given as input. It involves several pre-processing steps such as skull stripping, bias correction and segmentation. Once the segmentation is completed, three 2D slices (axial, coronnal, saggital) are extracted from the segmented MRI Image. In this manner the dataset is prepared which is then used for training a simple CNN. Here, the problem is approached in two methods. One method is to prepare a single dataset wherein the axial,coronnal and saggital images are present within each class which is then fed into a single classifier. Another method is to prepare three datasets (each contains only one kind of data) which is then fed into three classifers (Axial, Coronnal and Saggital Classifier). When compared, it was observed that the combined classifier (Axial,Coronnal,Saggital- **95%**) has more accuracy than the single classifier (**86%**)

## Data

In this project, we have obtained the brain MRI Images from **ADNI** website ([http://adni.loni.usc.edu/](http://adni.loni.usc.edu/)).

ADNI is an ongoing, multicenter cohort study, started from 2004. It focuses on understanding the diagnostic and predictive value of Alzheimers disease specific biomarkers. Our data included the data of 1,500 patients which resulted in a total of 1,80,000 images after augmentation.

### **Image preprocessing**
The images are preprocessed in order to produce the dataset and more detailed information can be found here ([here](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/preprocessing/description.md))

## Approaches used
### 1] Single Classifier
![Fig 1 - Single Classifier](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Images/Single_CLassifier_Flowchart.png)

The above figure shows the architecture for single classifer. Initially, the dataset is in 3D format(Nifti extension). The dataset is preprocessed to produce a 2D Image dataset wherein three folders are present- AD,CN, MCI. Each class has three kinds of data, ie, Axial, Coronnal and Saggital. In this manner, a single dataset is produced which is then fed into a Simple CNN Architecture. It was observed that the trained model achieved an accuracy of 86%. One method to improve the accuracy is to not mix up different kinds of data within a class as it makes the problem more complex. 

### 2] Combined Classifier

![Fig 2 - Combined_Classifer ](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Images/Multiple_Classifer_Architecture.png)

The above figure shows the architecture for multiple classifier. Here, the method is similar to the previous approach till the pre-processing phase.Once the dataset is produced, it is divided into three datasets so that each dataset consist of only one kind of data. In this manner, the complexity of the problem is reduced. Each of the dataset is fed into three seperate classifiers - Axial, Coronnal, Saggital(each of which is a Simple CNN Architecture).It was observed that the individual classifiers were able to acquire an accuracy of 94% each. Once the training is completed, the models are combined to give the final output.   

## Google Colab Notebooks
1] [Data_preprocessing.ipynb](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Jupyter_Notebooks/Data_preprocessing.ipynb)) is used to preprocess the ADNI Dataset. In this notebook, the packages for the same are mentioned.The user needs to only run ([this](will mention) file.

2] [Data_augmentation.ipynb](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Jupyter_Notebooks/Data_Augmentation.ipynb) is used to augment the preprocessed dataset. More information is available within the notebook.
