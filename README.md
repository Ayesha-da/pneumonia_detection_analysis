# Pneumonia Detection Analysis

A project by: Ayesha Shaheen, Chelsea Langford, Matthew Breitner, Nidhi Pandya, and William Johnson

## Roles and Responsibilities

### GitHub Repository Owner

Ayesha Shaheen

### Database Storage and Management

Matthew Breitner

### Database ETL and Analysis

Nidhi Pandya

### Machine Learning Model

Ayesha Shaheen

William Johnson

### JavaScript File

Nidhi Pandya

Ayesha Shaheen

### Visualizations and Presentation Development

Chelsea Langford (Dashboard creation using HTML, CSS, and Python)

Matthew Breitner (Tableau, Google Slides)

William Johnson (Google Slides)

## Project Overview

### Selected Topic

Our project will use neural networks and deep learning to interpret chest x-ray images and classify them as pneumonia or not-pneumonia/other.

### Project Use Case

- The use case for generating this model is to develop a process to objectively analyze and interpret x-ray images with a rate of speed and accuracy that is potentially better than the human eye.
- This model could be a valuable resource for doctors and students as another tool for interpreting x-ray images and validating/invalidating their personal diagnoses.
- Additionally, this model could be further applied to chest x-rays with diagnoses outside of pneumonia and be adapted to interpret the x-rays of patients with an array of diagnoses. Our goal is to generate a model that can predict the nuances between pneumonia, normal, or other to account that there are many potential diagnoses besides pneumonia to consider.

### Questions to Answer

- Can a machine learning model find distinct differences in chest x-ray images and accurately classify them as pneumonia or non-pneumonia/other?
- What level of accuracy can be achieved by this model? Can it be considered a reliable resource for individuals who are diagnosing patients?
- Long term, how can this model be applied to classifying other diagnoses and provide accurate predictions for diagnoses that are not pneumonia?
- Can this model be generated without introducing bias?

### Technologies Used

#### Database Technologies

- AWS for data storage
- PGAdmin and PostgreSQL for data table generation and manipulation
- Google Collab Notebooks for cloud database connection

#### Machine Learning Technologies

- Google Collab for machine learning model generation
- cv2 package for image analysis and preparation
- TensorFlow package for machine learning model generation

#### Visualization Technologies

- hvplot/plotly for visualizing the outputs of our classification model and results of our model
- Tableau Public for additional visualization support
- Heroku for hosting visualizations

## Pneumonia Statistics and Visualizations

### Pneumonia Statistics Data Sources

WHO: [https://www.who.int/health-topics/pneumonia#tab=tab_1](https://www.who.int/health-topics/pneumonia#tab=tab_1)

Our World In Data: [https://ourworldindata.org/pneumonia](https://ourworldindata.org/pneumonia)

CDC (general Pneumonia): [https://www.cdc.gov/dotw/pneumonia/index.html](https://www.cdc.gov/dotw/pneumonia/index.html)

CDC (US Data): [https://wonder.cdc.gov/controller/datarequest/D76;jsessionid=303DD855FA935405980D61135452](https://wonder.cdc.gov/controller/datarequest/D76;jsessionid=303DD855FA935405980D61135452)

### Visualizations

According to data collected by the likes of the CDC, NIH, WHO and other medical research organizations, pneumonia is a treatable disease if there is proper access to quick and efficient medical care. Although your chances of survival are incredibly high, we have found that pneumonia remains a major cause of death around the world. Looking at global statistics, pneumonia is especially deadly in countries with a low GDP per capita. This is likely due to not having the proper medical resources to avoid getting sick in the first place and the tools necessary for effective treatment. The following graph highlights the clear correlation between having a lower GDP and higher amount of pneumonia deaths. \*This data was collected from Bernadeta Dadonaite and Max Roser (2018) - "Pneumonia". Published online at OurWorldInData.org. Retrieved from: '[https://ourworldindata.org/pneumonia](https://ourworldindata.org/pneumonia)' [Online Resource]

- Date Range: 1990-2018 [https://public.tableau.com/app/profile/matthew.breitner/viz/DeathTotalsbyGDPperCapita/TotalDeathsperGDP2](https://public.tableau.com/app/profile/matthew.breitner/viz/DeathTotalsbyGDPperCapita/TotalDeathsperGDP2)

The following graph shows the trends in number of deaths in each country around the world. You can see that the countries with lower GDP, specifically in Central Africa and South/West Asia, follow the trend of lower GDP and higher deaths due to pneumonia. \*This data was collected from Bernadeta Dadonaite and Max Roser (2018) - "Pneumonia". Published online at OurWorldInData.org. Retrieved from: '[https://ourworldindata.org/pneumonia](https://ourworldindata.org/pneumonia)' [Online Resource] \*Date Range: 1990-2018 [https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalMortalityRatebyAge/GlobalMortalityRatebyAge](https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalMortalityRatebyAge/GlobalMortalityRatebyAge)

Pneumonia is especially deadly in children under the age of five in underdeveloped nations or countries with lower GDP. The below chart showcases the same global trends for the total # of deaths in children under five years old. \*This data was collected from Bernadeta Dadonaite and Max Roser (2018) - "Pneumonia". Published online at OurWorldInData.org. Retrieved from: '[https://ourworldindata.org/pneumonia](https://ourworldindata.org/pneumonia)' [Online Resource] \*Date Range: 1990-2018 [https://public.tableau.com/app/profile/matthew.breitner/viz/TotalChildDeathsbyCountryOverTime/ChildMortalityRatesbyCountrybyYear](https://public.tableau.com/app/profile/matthew.breitner/viz/TotalChildDeathsbyCountryOverTime/ChildMortalityRatesbyCountrybyYear)

Child wasting, a child who is too thin for their height, is the number one cause of pneumonia in children under five years old. Household pollution (e.g., non-proper ventilation, cleaning pollutants, smoke from cooking) is another major cause of pneumonia in children around the world. \*This data was collected from Bernadeta Dadonaite and Max Roser (2018) - "Pneumonia". Published online at OurWorldInData.org. Retrieved from: '[https://ourworldindata.org/pneumonia](https://ourworldindata.org/pneumonia)' [Online Resource] \*Date Range: 1990-2018 [https://public.tableau.com/app/profile/matthew.breitner/viz/CausesofPneumoniainChildrenUnderFiveYearsOld/GlobalChildMortalitybyRiskFactor2?publish=yes](https://public.tableau.com/app/profile/matthew.breitner/viz/CausesofPneumoniainChildrenUnderFiveYearsOld/GlobalChildMortalitybyRiskFactor2?publish=yes)

Globally, adults over the age of 70 years are also at a significantly higher risk of death from pneumonia in countries with lower GDP. The major causes of pneumonia in this demographic are not having proper access to handwashing facilities, pollution, smoking, and secondhand smoke. The following graph shows how these four causes have worsened for older adults in recent years. \*This data was collected from Bernadeta Dadonaite and Max Roser (2018) - "Pneumonia". Published online at OurWorldInData.org. Retrieved from: '[https://ourworldindata.org/pneumonia](https://ourworldindata.org/pneumonia)' [Online Resource] \*Date Range: 1990-2018 [https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalMortalityOver70byRiskFactorbyYear/GlobalMortalityOver70byRiskFactorbyYear](https://public.tableau.com/app/profile/matthew.breitner/viz/GlobalMortalityOver70byRiskFactorbyYear/GlobalMortalityOver70byRiskFactorbyYear)

\*Other data sources used for other analysis and visualizations not listed on the site.

1. Pneumonia Death Stats Source: [https://wonder.cdc.gov/controller/datarequest/D76;jsessionid=303DD855FA935405980D61135452](https://wonder.cdc.gov/controller/datarequest/D76;jsessionid=303DD855FA935405980D61135452)
2. Vaccination Stats: [https://www.kff.org/statedata/custom-state-report/?view=3&amp;i=32739~32172~444199&amp;g=us~al~ak~az~ar~ca~co~ct~de~dc~fl~ga~hi~id~il~in~ia~ks~ky~la~me~md~ma~mi~mn~ms~mo~mt~ne~nv~nh~nj~nm~ny~nc~nd~oh~ok~or~pa~ri~sc~sd~tn~tx~ut~vt~va~wa~wv~wi~wy](https://www.kff.org/statedata/custom-state-report/?view=3&amp;i=32739~32172~444199&amp;g=us~al~ak~az~ar~ca~co~ct~de~dc~fl~ga~hi~id~il~in~ia~ks~ky~la~me~md~ma~mi~mn~ms~mo~mt~ne~nv~nh~nj~nm~ny~nc~nd~oh~ok~or~pa~ri~sc~sd~tn~tx~ut~vt~va~wa~wv~wi~wy)
3. Hospitalization Rates: [https://pubmed.ncbi.nlm.nih.gov/29017956/](https://pubmed.ncbi.nlm.nih.gov/29017956/)

### Database ERD

#### Global Pneumonia Dataset

For our pneumonia US and global statistics, we will be hosting the data in PostgreSQL tables in PGAdmin. Our RDS is hosted on AWS so ultimately the data will live in the cloud. Please reference the following ERD and queries used to update the database for the global statistics.

![image](https://user-images.githubusercontent.com/84791455/140665154-49217357-a77b-41eb-ac35-5a1ff0be53c5.png)

[https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/data_ETL/ERD/global_statistics_table_queries.sql](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/data_ETL/ERD/global_statistics_table_queries.sql)

#### US Pneumonia dataset

![ERD for US](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/Presentation_ERD/ERDiagram.PNG)

We uploaded this data on AWS RDS using PGAdmin data engine. To create the database we used this [sql query](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/Presentation_ERD/presentation_createTable.sql).

This data was then used in Tableau for analysis

- [Deaths_by_age_group](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyAgeGroup/DeathsbyAgeGroup)
- [Deaths_by_gender](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyGender/DeathsbyGender)
- [Deaths_by_month](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyMonth/DeathsbyMonth)
- [Deaths_by_race](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyRace/DeathsbyRace)
- [Deaths_by_state_by_age_group](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyAgeGroup/DeathsbyStatebyAgeGroup)
- [Deaths_by_state_by_gender](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyGender/DeathsbyStatebyGender)
- [Deaths_by_state_by_race](https://public.tableau.com/app/profile/matthew.breitner/viz/PneumoniaDeathsbyStatebyRace/DeathbyStatebyRace)

## Machine Learning and Image Analysis

### Machine Learning Datasets


1. Chest X-Ray Images (Pneumonia)
  
  A. Source Link: [https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
  
  B. Source Format (quote from source): The dataset was originally organized into 3 folders (train, test, val) and contained subfolders for Pneumonia and Normal. There are 5,863 X-Ray images (JPEG) and 2 categories (Pneumonia/Normal).
  
  C. Source Details: "Chest X-ray images (anterior-posterior) were selected from retrospective cohorts of pediatric patients of one to five years old from Guangzhou Women and Children's Medical Center, Guangzhou. All chest X-ray imaging was performed as part of patients' routine clinical care... For the analysis of chest x-ray images, all chest radiographs were initially screened for quality control by removing all low quality or unreadable scans. The diagnoses for the images were then graded by two expert physicians before being cleared for training the AI system. To account for any grading errors, the evaluation set was also checked by a third expert."

  D. Notes: This was our initial dataset and serves as the basis for the intended format for other datasets. This set only identifies normal vs pneumonia, and does not consider other potential Lung issues that could also show up (cancer, covid, TB, etc.)

2. NIH Chest X-rays
  
  A. Source Link: [https://www.kaggle.com/nih-chest-xrays/data](https://www.kaggle.com/nih-chest-xrays/data)
  
  B. Source Format: Files were divided into 12 different folders with no differentiators in the folder or image names. There is a csv file which contains the labels for the images.

  C. Source Details (quote from source): "This NIH Chest X-ray Dataset is comprised of 112,120 X-ray images with disease labels from 30,805 unique patients. To create these labels, the authors used Natural Language Processing to text-mine disease classifications from the associated radiological reports. The labels are expected to be >90% accurate and suitable for weakly-supervised learning. The original radiology reports are not publicly available, but you can find more details on the labeling process in this Open Access paper: "ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases." (Wang et al.)"

  D. Notes: This is the largest of the datasets we reviewed, and it also has the most varied assortment of lung images, classifying 14 different diseases. Once we determine how we are building the buckets for our model, we'll need to format the files accordingly. The other concern with this dataset is that unlike the other sets, it was put together using their own NLP program that claims only a 90% accuracy, so while the original files are verified by radiologists, there is a larger margin of error due to the additional step of data mining.

#### Other Image Datasets considered but not used

RSNA Pneumonia Detection Challenge

Source Link: [https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge/overview)

Source Details (quote from source): "Chest X-Rays are the most commonly performed diagnostic imaging study. Several factors such as positioning of the patient and depth of inspiration can alter the appearance of the CXR [4], complicating interpretation further. In addition, clinicians are faced with reading high volumes of images every shift...To improve the efficiency and reach of diagnostic services, the Radiological Society of North America has reached out to Kaggle's machine learning community and collaborated with the US National Institutes of Health, The Society of Thoracic Radiology, and MD.ai to develop a rich dataset for this challenge. The RSNA is an international society of radiologists, medical physicists, and other medical professionals with more than 54,000 members from 146 countries across the globe. They see the potential for ML to automate initial detection (imaging screening) of potential pneumonia cases to prioritize and expedite their review."

Novel COVID-19 Chest Xray Repository

Source Link: [https://www.kaggle.com/tawsifurrahman/covid19-radiography-database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)

Source Details (quote from source): "A team of researchers from Qatar University, Doha, Qatar, and the University of Dhaka, Bangladesh along with their collaborators from Pakistan and Malaysia in collaboration with medical doctors have created a database of chest X-ray images for COVID-19 positive cases along with Normal and Viral Pneumonia images. This COVID-19, normal, and other lung infection dataset is released in stages. In the first release, we have released 219 COVID-19, 1341 normal, and 1345 viral pneumonia chest X-ray (CXR) images. In the first update, we have increased the COVID-19 class to 1200 CXR images. In the 2nd update, we have increased the database to 3616 COVID-19 positive cases along with 10,192 Normal, 6012 Lung Opacity (Non-COVID lung infection), and 1345 Viral Pneumonia images. We will continue to update this database as soon as we have new x-ray images for COVID-19 pneumonia patients."

### Database Set Up

We will be hosting our data on AWS using the S3 Buckets and a PostgreSQL RDS. Our dataset has over 5,000 images of chest x-rays that will be run through our machine learning model to determine if we can predict whether someone has Pneumonia. We chose to use AWS since it can easily store non-text data (images), our data is stored in the cloud so everyone can access it from their local devices, and we can upload our final data into a RDS for future querying and analysis.

### S3 Bucket Links

- [https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset](https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset)

### RDS Endpoint

- pneumonia-detection-analysis.cyhi4xykqawo.us-east-1.rds.amazonaws.com

### Data ETL

Our project is to detection pneumonia using chest x-ray images. We use Kaggle [dataset](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia). There are 5000+ x-ray images with 2 categories(Pneumonia/Normal)

![imag1](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/data_ETL/data_ETL/Resources/images/dataprocessing_concept02.png) 
We are using two datasets with the possibility of using more in the future.

1. Dataset1 - source from [Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)

This is a clean dataset with a folder structure:

- train
  - Pneumonia
  - Normal
- test
  - Pneumonia
  - Normal

1. Dataset2 - source from [Kaggle](https://www.kaggle.com/ingusterbets/nih-chest-x-rays-analysis)

This is not a clean dataset. Using [GetImages_fun](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/GetImages_Fun.ipynb) code, we are getting images that mimic Dataset1's folder structure. We decided to use 3 buckets (Pneumonia, Normal and Others) in dataset2, the code needs to be updated in the file to get the third bucket.(In progress)

we got the 3 buckets (Pneumonia, Normal and Others) from Dataset 2 and we used it in our Machine Learning model.

### Data Extraction and Transformation

We started to extract the data from our [s3 bucket](https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset?region=us-east-1&amp;prefix=chest_xray/&amp;showversions=false) and did some transformation in the Dataset1 images using [project_sample](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/data_ETL/project_sample.ipynb) code.

The Dataset's size is bigger and if all members of the team used the cloud images from the S3 bucket, the team would incur possible charges from AWS. To avoid this, we decided to store the images on our local machines to train and test the ML model.

Once we got our trained model, we created a Flask application and connected it with the model using [app.py](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/app.py) file. In this application when the user uploads an image it will store on [s3 bucket](https://s3.console.aws.amazon.com/s3/buckets/pneumoniadataset) and then it will be used for Prediction.

![img1](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/DataFlowDiagram.png)

### Machine Learning Model

![Conventional-machine-learning-vs-deep-learning](https://user-images.githubusercontent.com/84524153/138568406-ea33abaa-3e03-4d22-90e8-64034431f6df.png)

In Conventional Programming, decision making is based on IF-ELSE conditions. Therefore, many solutions cannot be modeled with it. One of the main reasons behind this is the variation of the input data variable, which increases the problem's complexity. On the contrary, machine learning programming solves the problem by modeling the data with train data and test data. Based on these data and statistical models, machine learning predicts the result.

In deep learning, we will be using a convolutional neural network (CNN/ConvNet), a class of deep neural networks, most commonly applied to analyze visual imagery. We are primarily working with images, and we need CNN model to take in these images, process them and give us the desired output by classifying them correctly as "Normal" or "Pneumonia".

- Preliminary Data Preprocessing

The image is read using cv2 and grayscale and is resized to 150,150 for easy processing.

![preprocessing](https://user-images.githubusercontent.com/84524153/140662909-526848c1-2732-4d86-a31b-3fd976cebf4b.png)

- Preliminary Feature Engineering

We also reshaped the X_train, X_test arrays and y_train, y_test arrays to use in Convolutional Neural Network Layers.

- Splitting Data Into Training and Testing Sets

Our data is imbalanced. To avoid this and overfitting, we performed data augmentation. The idea of data augmentation is to perform distortions to our existing data to decrease unnecessary variations on the data. For example, we apply horizontal flip, random zoom, height, and width shift to standardize the images for consistency and then we normalize the data, so it converges faster.

- Model choice, Limitations, and Benefits

Convolutional neural networks (CNN) are used for image classification and recognition because of its high accuracy. It was proposed by computer scientist Yann LeCun in the late 90s, when he was inspired from the human visual perception of recognizing things. They are one of the most popular models used today. This neural network computational model uses a variation of multilayer perceptrons and contains one or more convolutional layers that can be either entirely connected or pooled. These convolutional layers create feature maps that record a region of image which is ultimately broken into rectangles and sent out for nonlinear processing.

Benefits:

✓Very high accuracy in image recognition problems.

✓Automatically detects the important features without any human supervision.

✓Weight sharing.

Limitations:

✓CNN does not encode the position and orientation of object.

✓Lack of ability to be spatially invariant to the input data.

✓Lots of training data is required.

- Hardware / Software configuration

Google Compute Engine backend (GPU) and python 3.7, Spyder IDE(v 5.1.5), keras and TensorFlow on local computer.

For the model training we used google colab-GPU to run the model and process the charts. We encountered few issues while running model on local machine such as shutting down of kernel due to delayed times.

- Difficulties Encountered While Processing Images

We worked with over 5,800 images and encountered a very long training time, installation of new packages such as cvs and updating the version of TensorFlow. We also faced the problem of overfitting.

- CNN Model Architectures and Layers

We tried various hyperparameters such as different optimizers like "adam", "rmsprop", various activation functions such as sigmoid, relu and adding dropout layers, normalization function, and increasing/decreasing nodes.

Our model had an overfitting problem. We used early stopping criteria to stop training once the model performance stopped improving. We also added a "patience" argument that adds a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. We also tried to use the early stopping technique by adding and adjusting "baseline".

EarlyStopping(monitor='val_accuracy', mode='max') overfitting pneumonia class.
 EarlyStopping(monitor='val_loss', mode='min') overfitting normal class

EarlyStopping(monitor='val_loss', mode='min', patience=0.17) patience <= 0.17 overfit normal class and patience > 0.18 overfit pneumonia class.
 EarlyStopping(monitor='val_loss', mode='min', baseline=0.5) baseline <= 0.5 overfit pneumonia class and baseline > 0.5 overfits normal class.

We are also working on applying bias and keras regulizers to our model to improve accuracy.

#### Machine Learning Flowchart

![flow_chart_machine_model1](https://user-images.githubusercontent.com/84524153/141702747-5aafc0df-1446-47ff-8741-be4b38fa44ce.png)

- Graphs for the metrics and summary statistics

![accuracy](https://user-images.githubusercontent.com/84524153/141858509-480abdb0-123c-407d-92a8-b48e77d74349.png)
![loss](https://user-images.githubusercontent.com/84524153/141858541-757a5ffd-1d27-47c7-a40a-a9c8e29abe45.png)       
![confusion_matrix](https://user-images.githubusercontent.com/84524153/141858561-9e2a6d57-a9ce-4cc3-a738-f874bb03e8c8.png)
![roc_curve](https://user-images.githubusercontent.com/84524153/141858576-d90047c8-3e25-46f6-a14f-79ebeb56201a.png)

<img src= "https://user-images.githubusercontent.com/84524153/141858623-d04e4d03-e50d-4289-a2bb-809ba02a1483.png" width="800" />

- Model's Training Time

Google colab (GPU) took about 12 - 15 min to read the images. The training took about 20 - 25 min for 50 epochs.

#### Updating the Model
- Initially, our model was not performing well and was overfitting. We added a dropout layer and Batch Normalization to fight the issue. We made sure not to add a dropout layer in the first layer, in order to prevent our model from losing important features of the image.
- We still had an overfitting issue, so we tried the EarlyStopping technique using EarlyStopping callback to monitor performance measurements including validation_loss and validation_accuracy. Once triggered, the training process is stopped.
- EarlyStopping didn't help and we still had overfitting issues. The model was overfitting each class based on arguments such as setting patience to 0.17 and below fitting one class and setting patience above 0.17 overfitting another class.
- We trained our model using only a few images of both the classes from the dataset and that gave us few accurate results, but they were many wrong predictions too.
- Finally, we realized that our model is too complex, and it was not able to process new images. So we took out all the layers and started building the basic model of CNN and added layers until we observed the accuracy improvements level out. We tested our model on a validation set of images that the model had never seen before and now we have a model that gives about 84% accurate results.

#### Updating the Model From Segment 3 to Segment 4
  We trained the model to improve accuracy. We started with 25 epochs and the accuracy was 84%. Then we trained the model for 50 epochs, and we saw the accuracy improve significantly. When we trained the model for 100 epochs, we saw the accuracy drop. This made us realize the model started overfitting and so we went back to 50 epochs and now we have a model with accuracy of almost 90%.
  
The [code](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/trainmodel5.py) for the machine learning model is complete.

#### Limitations to Our Model

- What does the model do if it receives an image that is not an x-ray?                
  - Given more time and resources, we would work on a model that puts all other images in "Other" category.
- There could still be blind spots due to the limitations of our dataset.
- We could improve the model's output with the ability to upload and analyze multiple images at a time.

#### Other Model Options Considered

##### Pneumonia vs Other

- This model would analyze all images solely on whether they contained Pneumonia.
- It would group “Normal” images with results that contained “Other Diseases.”
- The major issue with this approach was that Pneumonia images would fall somewhere between “normal” and “other” and caused the model to skew everything into “Other” category.

##### Normal vs Pneumonia vs Other (2 model option)

- With this model, images would potentially be run through two different models using an If function.
- First model would compare Normal vs Everything Else.
- If classified as Normal, the model would return “Normal." Otherwise, it would push to the second model.
- The second model would compare Pneumonia to Other diseases.
- This model showed promise with preliminary results above 95%, however connection issues with running the model on our app locally and time constraints prevented it from being fully tested/deployed.

## Dashboard and Presentation

Our final dashboard will include a Google Slides presentation supplemented with images created in Tableau. Our interactive element will be a Heroku website that users can interact with to view visuals and information about our model and our model's output.

### Presentation

Google slides link:

[https://docs.google.com/presentation/d/1M43-xKBi9P2WS048Qo2mI5STj0Y7WPSW0KZxXJepePU/edit?usp=sharing](https://docs.google.com/presentation/d/1M43-xKBi9P2WS048Qo2mI5STj0Y7WPSW0KZxXJepePU/edit?usp=sharing)

PowerPoint backup of slides:

[https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/Pneumonia%20Analysis%20Presentation.pptx](https://github.com/Ayesha-da/pneumonia_detection_analysis/blob/main/Pneumonia%20Analysis%20Presentation.pptx)

### Heroku

Heroku Link: [https://pneumonia-detection-analysis.herokuapp.com/](https://pneumonia-detection-analysis.herokuapp.com/)

Interactive elements:

- Interact with our model! Upload file to receive a prediction of pneumonia vs. normal
- Filter Tableau visualizations
- View popups on visualizations for more details on data
- Select specific sections of the website to view based on homepage directory

### Presentation and Dashboard Technologies Used

- Google Slides
- Tableau Public
- Heroku for website hosting
- VS Code for HTML, CSS, JS, and Python editing
- Flask for website development
