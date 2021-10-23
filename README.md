# pneumonia_detection_analysis


Datasets Overview

1) Chest X-Ray Images (Pneumonia)
A. Source Link: https://www.kaggle.com/paultimothymooney/chest-xray-  pneumonia
B. Source Format: The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

C. Source Details: “Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children’s Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients’ routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert.”

D. Notes: the is our initial dataset and serves as the basis for the intended format for other datasets. This set only identifies normal vs pneumonia, and does not take into account other potential Lung issues that could also show up (cancer, covid, TB, etc) 

2) RSNA Pneumonia Detection Challenge
A. Source Link: https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview
B. Source Format: Images are broken up into two folders, test and train, but identifying data is in a separate (stage_2_detailed_class_info.csv) file that identifies images as either “Lung Opacity” or “No Lung Opacity/Not Normal” or “Normal” (Lung Opacity is the major visible symptom of
C. Source Details:  “Chest X-Rays are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR [4], complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift.

To improve the efficiency and reach of diagnostic services, the Radiological Society of North America has reached out to Kaggle’s machine learning community and collaborated with the US National Institutes of Health, The Society of Thoracic Radiology, and MD.ai to develop a rich dataset for this challenge.

The RSNA is an international society of radiologists, medical physicists and other medical professionals with more than 54,000 members from 146 countries across the globe. They see the potential for ML to automate initial detection (imaging screening) of potential pneumonia cases in order to prioritize and expedite their review.”
D. Notes: This dataset incorporates potential other lung issues and categorizes them based on the key symptom of Pneumonia (Lung Opacity). It does have three categories, the first two “Lung Opacity” and “Normal” are similar to our source set but also adds a third category that says “No Lung Opacity/Not Normal” to flag other potential issues and help the model differentiate when an image doesn’t fit into the two basic categories. It does not specify what those other issues might be, only that they exist and end users would need to further examine those images.       
