# Alzheimers Disease Detection Using Deep Learning

Alzheimer’s is a neurological disorder. The upcoming reports suggest it as the sixth leading cause of death. Normally MRI images  are analysed to identify whether the person has Alzheimer’s  or not. This method can be time consuming and might even lead to misdiagnosis. Hence we deployed Deep Learning for better accuracy in prediction.



## Abstract

This project aims to detect Alzheimer's Disease from the MRI scans ( in .nii extension) given as input. It involves several pre-processing steps such as skull stripping, bias correction and segmentation. Once the segmentation is completed, three 2D slices (axial, coronnal, saggital) are extracted from the segmented MRI Image. In this manner the dataset is prepared which is then used for training a simple CNN. Here, the problem is approached in two methods. One method is to prepare a single dataset wherein the axial,coronnal and saggital images are present within each class which is then fed into a single classifier. Another method is to prepare three datasets (each contain only one kind of data) which is then fed into three classifiers (Axial, Coronnal and Saggital). When compared, it was observed that the combined classifier(Axial,Coronnal,Saggital- **95%**) has more accuracy than the single classifier (**86%**).

## Data

In this project, we have obtained the brain MRI Images from **ADNI** website ([http://adni.loni.usc.edu/](http://adni.loni.usc.edu/)).

ADNI is an ongoing, multicenter cohort study, started from 2004. It focuses on understanding the diagnostic and predictive value of Alzheimers disease specific biomarkers. Our dataset included the data of 1,500 patients which resulted in a total of 1,80,000 images after augmentation.

#### Image preprocessing
The images are preprocessed to produce the dataset and more detailed information can be found [here](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/preprocessing/description.md)

## Approaches used
### 1] Single Classifier
![Fig 1 - Single Classifier](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Images/Single_Classifier_Flowchart.png)

The above figure shows the architecture for single classifer. Initially, the dataset is in 3D format(Nifti extension). The dataset is then preprocessed to produce a 2D Image dataset wherein three folders are present- AD,CN, MCI. Each class has three kinds of data, ie, Axial, Coronnal and Saggital. In this manner, a single dataset is produced which is then fed into a Simple CNN Architecture. It was observed that the trained model achieved an accuracy of 86%. One method to improve the accuracy is to not mix up different kinds of data within a class as it makes the problem more complex. 

### 2] Combined Classifier

![Fig 2 - Combined_Classifer ](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Images/Multiple_Classifer_Architecture.png)

The above figure shows the architecture for multiple classifier. Here, the method is similar to the previous approach till the pre-processing phase.Once the dataset is produced, it is divided into three datasets so that each dataset consist of only one kind of data. In this manner, the complexity of the problem is reduced. Each of the dataset is then fed into three seperate classifiers - Axial, Coronnal, Saggital(each of which is a Simple CNN Architecture).It was then observed, that the individual classifiers were able to acquire an accuracy of 94% each. Once the training is completed, the models are combined to give the final output.

## Google Colab Notebooks
1] [Data_preprocessing.ipynb](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Jupyter_Notebooks/Data_preprocessing.ipynb) is used to preprocess the ADNI Dataset. In this notebook, the packages for the same are mentioned.The user needs to only run [this](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/preprocessing/run_me.py) file in notebook.

2] [Data_augmentation.ipynb](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Jupyter_Notebooks/Data_Augmentation.ipynb) is used to augment the preprocessed dataset. More information is available within the notebook.

3] [Training_Single_Classifier.ipynb](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Jupyter_Notebooks/Training_Single_Classifier.ipynb) contains the details of training a simple CNN classifier with the preprocessed and augmented ADNI dataset.

4] [Training_Multiple_Classifier.ipynb](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Jupyter_Notebooks/Training_Multiple_CLassifier.ipynb) contains the details of training three CNN classifiers with the splitted, preprocessed and augmented ADNI dataset.

## Result
#### Graphs
During the initial phase of project, a small dataset was used which had 33,000 images and 10 tweaks were made in each model such as adding a dropout layer of 0.4, adding a dense layer of 512 neurons,etc to analyse the performance. In each of the tweak made, the best training accuracy, validation accuracy and loss was noted.This same process was repeated for each model while gradually increasing the data size and all these informations were consolidated into an [Excel Sheet](https://docs.google.com/spreadsheets/d/1h265xRbueSZ1y-vEKlDlhZAWuakgqYvdFgbRAzl1lR0/edit?usp=sharing). 

Out of the ten tweaks made, it was observed that the validation accuracy was highest in the case of adding a dropout layer of 0.4 or adding a dense layer of 512 neurons along with a dropout of 0.3 or 0.4.

In order to get a better understanding, six tweaks made in each model has been plotted against the datasize for each model and they are as follows :

#### 1] Combined Classifier
![Fig 3 - Combined_Classifier_Graph](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Graphs/Combined_Classifier_Graph.png)

Here, it can be observed that with the initial dataset, the accuaracy in every tweak remains within the range of 0.6-0.7 but when datasize is increased to 60k images, a sharp increase can be seen in the accuracy of every tweak. From the graph it can be concluded that, comparatively, the model with a dense layer of 1024 neurons has least performance while a dropout layer of 0.4 seems to be most effective in this case as it reaches to an accuracy of 0.86. 

#### 2] Axial Classifier
![Fig 4 - Axial_Classifier_Graph](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Graphs/Axial_Classifier_Graph.png)

In the above graph, the tweaks doesn't have a linear steady increase as there are many intersections between the lines. Initially, the tweaks have an accuracy in the range of 0.8-0.9 which is not the case of the model with a dense layer of 512 neurons but with the gradual increase in data size, it can be seen that the accuracy of every tweak increases and the best performance can be seen in the model with a dropout layer of 0.4.   

#### 3] Coronnal Classifier
![Fig 5 - Coronnal_Classifier_Graph](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Graphs/Coronnal_Classifier_Graph.png)

In this graph, initially every tweak has an accuracy in the range of 0.7-0.8 and the best performance can be observed in the models with the tweak of a dropout layer of 0.4 and a dense layer of 512 neurons along with a dropout layer of 0.4. Here it can also be observed that the model with a dense layer of 512 neurons have a performance similar to the model with a dense layer of 1024 neurons. In this graph, comparatively, the model with a dense layer of 512 neurons has the least performance which has an accuracy at 0.83

#### 4] Saggital Classifier
![Fig 5 - Saggital_Classifier_Graph](https://github.com/Lintaoommen/Alzheimers_Disease_Detection_Using_Deep_Learning/blob/master/Graphs/Saggital_Classifier_Graph.png)

Here, it can be observed that the tweaks doesn't have a lineaar steady increase and initially every tweak has an accuracy in the range of 0.80-0.85.It can also be observed that towards the end, every tweak achieves an accuracy in the range of 0.90-0.94 and the best performance can be observed in the model with a dense layer of 512 neurons along with a dropout layer of 0.4

## Challenges faced
Some of the problems that we ran into are as follows :
#### 1] Handling the data size
The data initially in 3D format was humongous to handle as 20 3D images itself add upto a size of 1 GB. So, this was solved by applying the preprocess pipeline on each input image and then deleting it and going for the next input.

#### 2] Finding the right model
Finding the right model was quite tricky and the data being complex, we tried fitting it with models such as MobileNet and ResNet but the performance wasn't satisfactory.

#### 3] Using the concept of transfer learning
During the initial phase of our project, we tried to use the concept of transfer learning with MobileNet but the problem with this is that MobileNet is pretrained with RGB Images and Medical Images are always in grayscale format. This was a major turning point for the project as it helped us to understand that we need to train a model from scratch and transfer learning can't be used.

## Scope
Currently, the model achieves as accuracy of 95% and some optimizations has already been performed such as:<br>
1] Pipelined the preprocessing functions.<br>
2] Reduced the first epoch training time by loading the dataset as zip file and not loading it from Gdrive via mounting.<br>
3] Split the dataset using validation split parameter of Image DataGenerator instead of using a code that manually move images from one folder to another.

##### Changes that can be made 
1] Now, the process of segmentation takes a lot of time to execute and it is quite time consuming. This problem can be solved by using more advanced methods such as [MONAI](https://monai.io/).<br>
2] Here, we have only taken grey matter, white matter and CSF (Cerebro spinal fluid) into consideration to detect AD. You can use other feautures such as Medial temperal lobe, etc too.

## Acknowledgement

#### Collaborators
1] [Silpa Chandran](https://www.linkedin.com/in/silpa-chandran-747a78182/)<br>
2] [Vidhya L Prathapan](https://www.linkedin.com/in/vidhya-l-prathapan-28282a180/)<br> 
3] [Krishnapriya P](https://www.linkedin.com/in/manschaftg-ap-ab78351b2/) 

#### Mentor
[Sleeba Paul](https://sleebapaul.github.io/)
