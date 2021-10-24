# Pneumonia Detection Analysis
A project by: Ayesha Shaheen, Chelsea Langford, Matthew Breitner, Nidhi Pandya, and William Johnson

## Project Overview
### Selected Topic
Our project will use neural networks and deep learning to interpret chest x-ray images and classify them as pneumonia, not pneumonia, or other.
### Project Use Case
- The use case for generating this model is to develop a process to objectively analyze and interpret x-ray images with a rate of accuracy that is potentially better than the human eye.
- This model could be a valuable resource for doctors and students as another tool for interpreting x-ray images and validating/invalidating their personal diagnoses.
- Additionally, this model could be futher applied to chest x-rays with diagnoses outside of pneumonia and be adapted to interpret the x-rays of patients with an array of diagnoses. Our goal is to generate a model that can predict the nuances between pneumonia, normal, or other to account that there are many potential diagnoses besides pneumonia to consider.
 
### Questions to Answer
- Can a machine learning model find distinct differences in chest x-ray images and accurately classify them as pneumonia, normal, or other?
- What level of accuracy can be achieved by this model? Can it be considered a reliable resource for individuals who are diagnosing patients?
- Long term, how can this model be applied to classifying other diagnoses based on x-ray image analysis?
- Can this model be generated without introducing bias? 

## Dataset Description 
### Main Dataset
Our primary dataset is an image library sourced from this [Kaggle dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). The kaggle dataset was originally sourced from [this article](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5#relatedArticles) in Cell, which conducted an analysis of identifying medical diagnoses using image-based deep learning. The image library consists of chest x-ray images that have been classfied as normal, baterial pneumonia, or viral pneumonia.
### Additional Datasets to Consider
- Our project team is considering the addition of other chest x-ray datasets to avoid introducing bias into our model. We are considering the implications of using x-ray images from a single source, which could cause our model to only correctly classify the image if provided by the same source as the original dataset.
- A second dataset we may introduce into our model is [this x-ray image library](https://data.mendeley.com/datasets/jctsfj2sfn/1) from Mendeley Data, which includes images with diagnoses classified as COVID-19, pneumonia, or normal.
- We are in the process of sourcing other datasets to consider beyond this, such as additional libraries of "normal" chest x-ray images and/or libraries of chest x-rays with other diagnoses. We hypothesize that the greater variety of image sources we injest into our model, the more accurate and less biased the output will be.
- Beyond the datasets required to inform our model, we can also introduce demographic data for pneumonia and COVID-19 diagnoses rates by location and patient descriptions. While not essential to the success of our model, this information will allow us to further refine the scope and implications of our model's use case and development long term.

## Datasets Overview

### 1) Chest X-Ray Images (Pneumonia)

#### A. Source Link: 
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

#### B. Source Format: 

The dataset is organized into 3 folders (train, test, val) and contains subfolders for each image category (Pneumonia/Normal). There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).

#### C. Source Details: 

"Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients' routine clinical care.

For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert."

#### D. Notes: 

This our initial dataset and serves as the basis for the intended format for other datasets. This set only identifies normal vs pneumonia, and does not take into account other potential Lung issues that could also show up (cancer, covid, TB, etc) 

### 2) RSNA Pneumonia Detection Challenge

#### A. Source Link: 

https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview

#### B. Source Format: 

Images are broken up into two folders, test and train, but identifying data is in a separate (stage_2_detailed_class_info.csv) file that identifies images as either "Lung Opacity" or "No Lung Opacity/Not Normal" or "Normal" (Lung Opacity is the major visible symptom of

#### C. Source Details:  

"Chest X-Rays are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR [4], complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift.

To improve the efficiency and reach of diagnostic services, the Radiological Society of North America has reached out to Kaggle's machine learning community and collaborated with the US National Institutes of Health, The Society of Thoracic Radiology, and MD.ai to develop a rich dataset for this challenge.

The RSNA is an international society of radiologists, medical physicists and other medical professionals with more than 54,000 members from 146 countries across the globe. They see the potential for ML to automate initial detection (imaging screening) of potential pneumonia cases in order to prioritize and expedite their review."

#### D. Notes: 

This dataset incorporates potential other lung issues and categorizes them based on the key symptom of Pneumonia (Lung Opacity). It does have three categories, the first two "Lung Opacity" and "Normal" are similar to our source set but also adds a third category that says "No Lung Opacity/Not Normal" to flag other potential issues and help the model differentiate when an image doesn't fit into the two basic categories. It does not specify what those other issues might be, only that they exist and end users would need to further examine those images. 

### 3. Novel COVID-19 Chestxray Repository  

#### A. Source Link: 

https://www.kaggle.com/tawsifurrahman/covid19-radiography-database

#### B. Source Format: 

Files are split into 4 folders labeled as either COVID, Lung_Opacity, Normal, or Viral Pneumonia. Individual files also contain these labels in the filename. 

#### C: Source Details: 

"A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal, and other lung infection dataset is released in stages. In the first release, we have released 219 COVID-19, 1341 normal, and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients."  

#### D. Notes

Caveat for this data is that the "Viral Pneumonia" folder contains images taken exclusively from our original dataset so it can be ignored. The remaining files will just need to be formatted to fit whatever identifiers we determine we are going to use. 

### 4. NIH Chest X-rays

#### A. Source Link:

https://www.kaggle.com/nih-chest-xrays/data

#### B. Source Format: Files are divided into 12 different folders with no differentiators in the folder or image names. There is a csv file which contains the labels for the images. 


#### C. Source Details:

"This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.)"

#### D. Notes:

This is the largest of the datasets that we have so far and it also has the most varied assortment of lung images, classifying 14 different diseases. Once we determine how we are building the buckets for our model, we'll need to format the files accordingly. The other concern with this dataset is that unlike the other sets, it was put together using their own NLP program that claims only a 90% accuracy, so while the original files are verified by radiologists, there is a larger margin of error due to the additional step of data mining. 



## Technologies Used
### Database Technologies
- AWS for data storage
- PGAdmin and PostgreSQL for data table generation and manipulation
- Google Collab Notebooks for cloud database connection
### Machine Learning Technologies
- Google Collab for machine learning model generation
- cv2 package for image analysis and preparation
- TensorFlow package for machine learning model generation
### Visualization Technologies
- hvplot/plotly for visualizing the outputs of our classification model and results of our model
- Tableau Public for additional visualization support
- Heroku for hosting visualizations

## Database Set Up
We will be hosting our data on AWS through the use of the S3 Buckets and the a postgreSQL RDS. Our dataset has over 5,000 images of chest x-rays that will be run through our machine learning model to determine if we can predict whether or not someone has Pneumonia. We chose to use AWS since it can easily store non-text data (images), our data is stored in the cloud so everyone can access it from their local devices, and we can upload our final data into a RDS for future querying and analysis. 

### S3 Bucket Links
- Test: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=test/&showversions=false
    - Normal: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=test/NORMAL/&showversions=false
    - Pneumonia: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=test/PNEUMONIA/&showversions=false

- Train: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=train/&showversions=false
    - Normal: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=train/NORMAL/&showversions=false
    - Pneumonia: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=train/PNEUMONIA/&showversions=false

- Val: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=val/&showversions=false
    - Normal: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=val/NORMAL/&showversions=false
    - Pneumonia: https://s3.console.aws.amazon.com/s3/buckets/pneumonia-detection-analysis?region=us-east-1&prefix=val/PNEUMONIA/&showversions=false

### RDS Endpoint
- pneumonia-detection-analysis.cyhi4xykqawo.us-east-1.rds.amazonaws.com

# Machine Learning
![Conventional-machine-learning-vs-deep-learning](https://user-images.githubusercontent.com/84524153/138568406-ea33abaa-3e03-4d22-90e8-64034431f6df.png)





In Conventional Programming, decision making is based on IF-ELSE conditions. Therefore, many solutions cannot be modeled with it. One of the main reasons behind this is the variation of the input data variable, which increases the problem’s complexity. On the contrary, machine learning programming solves the problem by modeling the data with train data and test data. Based on these data and statistical models, machine learning predicts the result.

In deep learning, we will be using a convolutional neural network (CNN/ConvNet), a class of deep neural networks, most commonly applied to analyze visual imagery.We are primarily working with images and we need CNN model to take in these images, process them and give us the desired output by classifying them correctly as “normal” or “pneumonia”.

