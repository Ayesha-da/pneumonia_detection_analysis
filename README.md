
# pneumonia_detection_analysis
* Hardware or software configuration
   Google Compute Engine backend (GPU)
   For the model training we used google colab-GPU to run the model and process the charts. We encountered few issues while running model on local machine such as shutting down    of kernel due to delayed times.

* Did you experience any memory or other difficulties with the image or data size?
   
  Severe Overfitting, very long training time, installing new packages such as cvs and updating version of tensorflow.

* What does the model do if it receives an image that is not an x-ray (for example, a dog)?

  We are working on a model that will put all other images in “Other” category and it is still in testing phase  but due to time constrain of the project , it is not included as   for now.

* Did you consider or explore other CNN model architectures and other layers?

  We tried various hyperparameters such as different optimizers such as "adam", "rmsprop", variuos activation functions such as sigmoid, relu and adding dropout                    layers,normalization function, increasing/decreasing nodes ,to work with the model.
 
 Our model had overfitting problem. 
 The early stopping criteria would also help avoid overfitting  stop training once the model performance stops improving 

  We can account for this by adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. This can be done by setting the           “patience” argument. and baseline=0.5

  EarlyStopping(monitor='val_accuracy', mode='max') overfitting 0.5 above
  
  EarlyStopping(monitor='val_loss', mode='min') overfitting 0.5 below class

  EarlyStopping(monitor='val_loss', mode='min', baseline=0.5)    baseline= 0.5 and below overfits pneumonia and above 0.5 overfits normal
  EarlyStopping(monitor='val_loss', mode='min', patience=0.17) patience=0.17 and below over fits normal, above 0.18 and above overfitting pneumonia
 
  We are also working on applying bias and keras regularizers to our model to improve accuracy. 

* Can you display graphs for the metrics over the training epochs (train loss, test loss, etc.)? What were the final numbers?

![accuracy](https://user-images.githubusercontent.com/84524153/141501445-dbedd357-2e8b-4db7-8a37-bc012eaaece0.png)
![loss](https://user-images.githubusercontent.com/84524153/141501481-db970c47-01f1-4991-8312-e38083c04727.png)
![confusion_matrix](https://user-images.githubusercontent.com/84524153/141501528-ca5d16e5-b9d4-41f6-a4a8-66713b7c7727.png)
![roc_curve](https://user-images.githubusercontent.com/84524153/141501497-1faa0665-7e95-4dcf-8d5b-826958bae9ac.png)
<img src= "https://user-images.githubusercontent.com/84524153/141504032-70eb93f8-a604-4671-89d0-5f21af2b51f6.png" width="800" />

* How long did the training take?

  Google colab (GPU) took about 15-20 min to read the images
  The training took about 20 - 25 min for 50 epochs .
  
  Future work
  Limitation in our model

Our model seems to have high recall and AUC score, but there could still be blindspots due to the limitations of dataset. Given more time and resources, I’d love to explore the following:
1. Include more diversity in the dataset in terms of patient residence and age


