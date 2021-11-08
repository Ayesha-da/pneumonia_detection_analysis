# Pneumonia Detection Analysis
A project by: Ayesha Shaheen, Chelsea Langford, Matthew Breitner, Nidhi Pandya, and William Johnson

## Roles and Responsibilities
#### GitHub Repository Owner
Ayesha Shaheen

#### Database Storage and Management
Matthew Breitner

#### Database ETL and Analysis
Nidhi Pandya

#### Machine Learning Model
Ayesha Shaheen

William Johnson

#### JavaScript File
Nidhi Pandya 

Ayesha Shaheen

#### Visualizations and Presentation Development
Chelsea Langford (Dashboard creation using HTML, CSS, and Python)

Matthew Breitner (Tableau, Google Slides)

William Johnson (Google Slides)


## Communication Protocol 
To manage team progress throughout the project timeline, our team has established the following communication process:
- Slack group which includes all team members 
- Daily check-ins on Slack
- Recurring meeting on Mondays and Wednesdays at 7PM in addition to our Tues/Thurs classes
- Additional meetings are scheduled as needed based on Slack discussions

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

## Dataset Descriptions 

### Machine Learning Datasets

Our primary Machine Learning datasets are image libraries sourced from Kaggle. The first dataset was originally sourced from [this article](https://www.cell.com/cell/fulltext/S0092-8674(18)30154-5#relatedArticles) in Cell, which conducted an analysis of identifying medical diagnoses using image-based deep learning. The image library consists of chest x-ray images that have been classfied as normal, baterial pneumonia, or viral pneumonia.

- Beyond the datasets required to inform our model, we can also introduce demographic data for pneumonia and COVID-19 diagnoses rates by location and patient descriptions. While not essential to the success of our model, this information will allow us to further refine the scope and implications of our model's use case and development long term.

#### Datasets Overview
 1) Chest X-Ray Images (Pneumonia)
  
  A. Source Link: 
   https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
 
  B. Source Format: 
   
   The dataset was originally organized into 3 folders (train, test, val) and contained subfolders for Pneumonia and Normal. There are 5,863 X-Ray images (JPEG) and 2   categories (Pneumonia/Normal).
  
  C. Source Details: 
   
   "Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients' routine clinical care.
  
   For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. In order to account for any grading errors, the evaluation set was also checked by a third expert."
  
  D. Notes: 
    
    This was our initial dataset and serves as the basis for the intended format for other datasets. This set only identifies normal vs pneumonia, and does not take into account other potential Lung issues that could also show up (cancer, covid, TB, etc) 

 2) NIH Chest X-rays
  
  A. Source Link:
  
  https://www.kaggle.com/nih-chest-xrays/data
  
  B. Source Format: 
  
  Files were divided into 12 different folders with no differentiators in the folder or image names. There is a csv file which contains the labels for the images. 
  
  C. Source Details:
  
    "This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.)"
  
   D. Notes:
  
    This is the largest of the datasets we reviewed and it also has the most varied assortment of lung images, classifying 14 different diseases. Once we determine how we are building the buckets for our model, we'll need to format the files accordingly. The other concern with this dataset is that unlike the other sets, it was put together using their own NLP program that claims only a 90% accuracy, so while the original files are verified by radiologists, there is a larger margin of error due to the additional step of data mining. 

 Other Image Datasets considered but not used

 RSNA Pneumonia Detection Challenge
  
  Source Link: 
   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview
  
  Source Details:  
  
   "Chest X-Rays are the most commonly performed diagnostic imaging study. A number of factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR [4], complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift.
   
    To improve the efficiency and reach of diagnostic services, the Radiological Society of North America has reached out to Kaggle's machine learning community and collaborated with the US National Institutes of Health, The Society of Thoracic Radiology, and MD.ai to develop a rich dataset for this challenge.
    
    The RSNA is an international society of radiologists, medical physicists and other medical professionals with more than 54,000 members from 146 countries across the globe. They see the potential for ML to automate initial detection (imaging screening) of potential pneumonia cases in order to prioritize and expedite their review."

Novel COVID-19 Chestxray Repository  
   
   Source Link: 
    https://www.kaggle.com/tawsifurrahman/covid19-radiography-database
   
   Source Details: 
    
    "A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal, and other lung infection dataset is released in stages. In the first release, we have released 219 COVID-19, 1341 normal, and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients."  

#### Pneumonia Statistics Data Sources

WHO: https://www.who.int/health-topics/pneumonia#tab=tab_1
Our World In Data: https://ourworldindata.org/pneumonia
CDC (general Pneumonia): https://www.cdc.gov/dotw/pneumonia/index.html
CDC (US Data): https://wonder.cdc.gov/controller/datarequest/D76;jsessionid=303DD855FA935405980D61135452



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
- https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset
### RDS Endpoint
- pneumonia-detection-analysis.cyhi4xykqawo.us-east-1.rds.amazonaws.com

## Data ETL
Our project is to detection pneumonia using chest x-ray images. We use Kaggle [dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). There are 5000+ x-ray images with 2 categories(Pneumonia/Normal)

![imag1](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/data_ETL/data_ETL/Resources/images/dataprocessing_concept02.png)
We are using two datasets with the possibility of using more in the future.

1. Dataset1 - source from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

This is a clean dataset with a folder structure:
 * train
   * Pneumonia
   * Normal
 * test
   * Pneumonia
   * Normal
  
2. Dataset2  - source from [Kaggle](https://www.kaggle.com/ingusterbets/nih-chest-x-rays-analysis) 

This is not a clean dataset. Using [GetImages_fun](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/GetImages_Fun.ipynb) code, we are getting images that mimic Dataset1's folder structure. We decided to use 3 buckets (Pneumonia, Normal and Others) in dataset2, the code needs to be updated in the file to get the third bucket.(In progress)

we got the 3 buckets (Pneumonia, Normal and Others) from Dataset 2 and we used it in our Machine Learning model.

#### Data Extraction and Trasformation: 

we started to extract the data from our [s3 bucket](https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset?region=us-east-1&prefix=chest_xray/&showversions=false) and did some trasformation in the Dataset1 images using [project_sample](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/project_sample.ipynb) code.

The Dataset's size is bigger and if all members of the team used the cloud images from the S3 bucket, the team would incur possible charges from AWS.  To avoid this, we decided to store the images on our local machines in order to train and test the ML model.

Once we got our trained model we created a Flask application and connected it with the model using [app.py](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/app.py) file. In this application when the user uploads an image it will store on [s3 bucket](https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset) and then it will be used for Prediction.

#### Visualization
For the visualization we used different data for analysis. The two datasets that we used were US Pneumonia dataset and Global Pneumonia dataset.

##### US Pneumonia dataset

![ERD for US](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/Presentation_ERD/ERDiagram.PNG)

we uploaded this data on AWS RDS using pgAdmin dataengine.
To create the database we used the [sql query](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/Presentation_ERD/presentation_createTable.sql)

This data was then used in tableu for analysis

* [Deaths_by_age_group](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyAgeGroup/DeathsbyAgeGroup)

* [Deaths_by_gender](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyGender/DeathsbyGender)

* [Deaths_by_month](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyMonth/DeathsbyMonth)
 
* [Deaths_by_race](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyRace/DeathsbyRace)
 
* [Deaths_by_state_by_age_group](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyAgeGroup/DeathsbyStatebyAgeGroup)

* [Deaths_by_state_by_gender](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyGender/DeathsbyStatebyGender)

* [Deaths_by_state_by_race](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyRace/DeathbyStatebyRace)


## Visualizations
Looking at data collected by the CDC, NIH, WHO, and other medical research organizations, we have determined that although Pneumonia is a treatable disease, it is still a major problem throughout much of the world and even the United States. 

Looking at the global statistics, we identified that pneumonia majorly effects children under the age of 5 and adults over the age of 70. Pneumonia is especially deadly for these age groups in countries that have a smaller GDP per capita as reference by the below chart. 

https://public.tableau.com/app/profile/matthew.breitner/viz/TotalDeathsperGDP/TotalDeathsperGDP

Because these countries have a lower GDP, they have less resources to combat Pneumonia from becoming a deadly disease. The following two graphics visualize the leading causes of Pneumonia in Children under the age of 5 and adults over the age of 70. 

https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalChildMortalitybyRiskFactor/GlobalChildMortalitybyRiskFactor

https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalMortalityOver70byRiskFactorbyYear/GlobalMortalityOver70byRiskFactorbyYear

Child mortality rates begin to increase in Africa and South/West Asia, which unfortunately follows the lower GDP trend. 

https://public.tableau.com/app/profile/matthew.breitner/viz/ChildMortalityRatesbyCountrybyYear/ChildMortalityRatesbyCountrybyYear

You can see this trend continue when looking at the overal mortality rates across the world. 

https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalMortalityRatebyAge/GlobalMortalityRatebyAge

We then analyzed to see if the same trends could be found in the United States. Although we did not see the same trends in terms of GDP (the largest states still had the highest death rates) we did see a similar trend in higher rates of death in adults over the age of 70. The following graphics outline the amount of deaths by age group and the amount of deaths by state within the US. 

https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyAgeGroup/DeathsbyAgeGroup
https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyAgeGroup/DeathsbyStatebyAgeGroup

Although medical technologies has advanced tenfold over the last 20 years, that has come as both an advantage and disadvantage for medical professionals around the world, specifcially radiologists. According to the Mayo clinic, the number of X-ray scans a radiologist have to anlyze in a given shift has gone from 1 image every 20 seconds to 1 image every 4 seconds. Now it is advantageous to scan so many images quickly but it makes it more difficult to accurately diagnose an image. This is were our model with come into play because it will not only assist in the increased speed of image reading but it can more accurately diagnose the image for pneumonia compared to theh naked eye. 

![Pneumonia_Image_Speed](https://user-images.githubusercontent.com/84791455/140665486-4de6a6d7-d99f-46ab-bb67-96a35ab41ea9.PNG)



### Database ERD 
For our pneumonia US and global statistics we will be hosting the data in postgresql tables in PGAdmin. Our RDS is hosted on AWS so ultimately the data will live in the cloud. Please reference the following ERD and queries used to update the database for the global statistics. 

![image](https://user-images.githubusercontent.com/84791455/140665154-49217357-a77b-41eb-ac35-5a1ff0be53c5.png)

https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/data_ETL/ERD/global_statistics_table_queries.sql


## Machine Learning
![Conventional-machine-learning-vs-deep-learning](https://user-images.githubusercontent.com/84524153/138568406-ea33abaa-3e03-4d22-90e8-64034431f6df.png)

In Conventional Programming, decision making is based on IF-ELSE conditions. Therefore, many solutions cannot be modeled with it. One of the main reasons behind this is the variation of the input data variable, which increases the problem’s complexity. On the contrary, machine learning programming solves the problem by modeling the data with train data and test data. Based on these data and statistical models, machine learning predicts the result.

In deep learning, we will be using a convolutional neural network (CNN/ConvNet), a class of deep neural networks, most commonly applied to analyze visual imagery.We are primarily working with images and we need CNN model to take in these images, process them and give us the desired output by classifying them correctly as “Normal” or “Pneumonia”.
The [code](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/trainmodel.py) for machine learning is complete.
- preliminary data preprocessing

  The image is read using cv2 and grayscale and is resized to 150,150 for easy processing.
 
![preprocessing](https://user-images.githubusercontent.com/84524153/140662909-526848c1-2732-4d86-a31b-3fd976cebf4b.png)
   
- preliminary feature engineering

   We also reshaped the X_train, X_test arrays and y_train, y_test arrays in order to use in Convolutional Neural Network Layers.
   
- Description of how data was split into training and testing sets 

   Our data is imbalanced. To avoid this and overfitting, we performed data augmentation. The idea of data augmentation is we perform some distortions to our existing data and      we get new various data. For example we apply horizontal flip, random zoom, height and width shift and then we normalize the data so it converges faster.
   
- Explanation of model choice, limitations and benefits.

  Convolutional neural networks (CNN) are used for image classification and recognition because of its high accuracy.It was proposed by computer scientist Yann LeCun in the late   90s, when he was inspired from the human visual perception of recognizing things.They are one of the most popular models used today. This neural network computational model     uses a variation of multilayer perceptrons and contains one or more convolutional layers that can be either entirely connected or pooled. These convolutional layers create       feature maps that record a region of image which is ultimately broken into rectangles and sent out for nonlinear processing.
  
  Benefits:
  
   ✓ Very High accuracy in image recognition problems.
 
   ✓ Automatically detects the important features without any human supervision.
 
   ✓ Weight sharing.

   Limitations:

   ✓ CNN do not encode the position and orientation of object.
 
   ✓ Lack of ability to be spatially invariant to the input data.
 
   ✓ Lots of training data is required.
  

## Dashboard
Our final dashboard will include a Google Slides presentation supplemented with images created in Tableau. Our interactive element will be a Heroku website that users can interact with to view visuals and information about our model and our model's output. 

#### Presentation
Google slides link:

https://docs.google.com/presentation/d/1M43-xKBi9P2WS048Qo2mI5STj0Y7WPSW0KZxXJepePU/edit?usp=sharing

Powerpoint backup of slides:

https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/Pneumonia%20Analysis%20Presentation.pptx

#### Heroku
Heroku Link: https://pneumonia-detection-analysis.herokuapp.com/

Interactive elements:
- Interact with our model! Upload file to receive a prediction of pneumonia vs. normal
- Filter Tableau visualizations 
- View popups on visualizations for more details on data
- Select specific sections of the website to view based on homepage directory

#### Dashboard Technologies Used
- Google Slides
- Tableau Public
- Heroku for website hosting
- VS Code for HTML, CSS, JS, and Python editing 
- Flask for website development

