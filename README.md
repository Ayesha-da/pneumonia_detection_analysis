# pneumonia_detection_analysis


## Datasets Overview

### 1) Chest X-Ray Images (Pneumonia)

#### A. Source Link: 
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

#### B. Source Format: 

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

#### C. Source Details: 

“Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.”

#### D. Notes: 

This our initial dataset and serves as the basis for the intended format for other datasets. This set only identifies normal vs pneumonia, and does not take into account other potential Lung issues that could also show up (cancer, covid, TB, etc) 

### 2) RSNA Pneumonia Detection Challenge

#### A. Source Link: 

https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview

#### B. Source Format: 

Images are broken up into two folders, test and train, but identifying data is in a separate (stage_2_detailed_class_info.csv) file that identifies images as either “Lung Opacity” or “No Lung Opacity/Not Normal” or “Normal” (Lung Opacity is the major visible symptom of

#### C. Source Details:  

“Chest X-Rays are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR [4], complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift.

To improve the efficiency and reach of diagnostic services, the Radiological Society of North America has reached out to Kaggle’s machine learning community and collaborated with the US National Institutes of Health, The Society of Thoracic Radiology, and MD.ai to develop a rich dataset for this challenge.

The RSNA is an international society of radiologists, medical physicists and other medical professionals with more than 54,000 members from 146 countries across the globe. They see the potential for ML to automate initial detection (imaging screening) of potential pneumonia cases in order to prioritize and expedite their review.”

#### D. Notes: 

This dataset incorporates potential other lung issues and categorizes them based on the key symptom of Pneumonia (Lung Opacity). It does have three categories, the first two “Lung Opacity” and “Normal” are similar to our source set but also adds a third category that says “No Lung Opacity/Not Normal” to flag other potential issues and help the model differentiate when an image doesn’t fit into the two basic categories. It does not specify what those other issues might be, only that they exist and end users would need to further examine those images. 

### 3. Novel COVID-19 Chestxray Repository  

#### A. Source Link: 

https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

#### B. Source Format: 

Files are split into 4 folders labeled as either COVID, Lung_Opacity, Normal, or Viral Pneumonia. Individual files also contain these labels in the filename. 

#### C: Source Details: 

“A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal, and other lung infection dataset is released in stages. In the first release, we have released 219 COVID-19, 1341 normal, and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients.”  

#### D. Notes

Caveat for this data is that the “Viral Pneumonia” folder contains images taken exclusively from our original dataset so it can be ignored. The remaining files will just need to be formatted to fit whatever identifiers we determine we are going to use. 

### 4. NIH Chest X-rays

#### A. Source Link:

https://www.kaggle.com/nih-chest-xrays/data

#### B. Source Format: Files are divided into 12 different folders with no differentiators in the folder or image names. There is a csv file which contains the labels for the images. 


#### C. Source Details:

“This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.)”

#### D. Notes:

This is the largest of the datasets that we have so far and it also has the most varied assortment of lung images, classifying 14 different diseases. Once we determine how we are building the buckets for our model, we’ll need to format the files accordingly. The other concern with this dataset is that unlike the other sets, it was put together using their own NLP program that claims only a 90% accuracy, so while the original files are verified by radiologists, there is a larger margin of error due to the additional step of data mining. 




