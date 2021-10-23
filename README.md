
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

# Machine Learning
![Conventional-machine-learning-vs-deep-learning](https://user-images.githubusercontent.com/84524153/138568406-ea33abaa-3e03-4d22-90e8-64034431f6df.png)





In Conventional Programming, decision making is based on IF-ELSE conditions. Therefore, many solutions cannot be modeled with it. One of the main reasons behind this is the variation of the input data variable, which increases the problem’s complexity. On the contrary, machine learning programming solves the problem by modeling the data with train data and test data. Based on these data and statistical models, machine learning predicts the result.

In deep learning, we will be using a convolutional neural network (CNN/ConvNet), a class of deep neural networks, most commonly applied to analyze visual imagery.We are primarily working with images and we need CNN model to take in these images, process them and give us the desired output by classifying them correctly as “normal” or “pneumonia”.
 
 
