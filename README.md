
# pneumonia_detection_analysis
* Hardware or software configuration
   Google Compute Engine backend (GPU)
   For the model training we used google colab-GPU to run the model and process the charts. We encountered few issues while running model on local machine such as shutting down    of kernel due to delayed times.

* Did you experience any memory or other difficulties with the image or data size?
   
  Severe Overfitting
 very long training time, installing new packages such as cvs and updating version of tensorflow.

* What does the model do if it receives an image that is not an x-ray (for example, a dog)?

We are working on a model that will put all other images in “Other” category and it is still in testing phase  but due to time constrain of the project , it is not included as fpr now.

* Did you consider or explore other CNN model architectures and other layers?

 We tried various hyperparameters such as different optimizers such as "adam", "rmsprop", variuos activation functions such as sigmoid, relu and adding dropout layers,normalization function, increasing/decreasing nodes ,to work with the model.
 
* Can you display graphs for the metrics over the training epochs (train loss, test loss, etc.)? What were the final numbers?



* How long did the training take?
Google colab (GPU) took about 15-20 min to read the images
The training took about 20 - 25 min for 50 epochs .

Our model had overfitting problem. 
The early stopping criteria would also help avoid overfitting  stop training once the model performance stops improving 


We can account for this by adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. This can be done by setting the “patience” argument. and baseline=0.5

early_stopping = EarlyStopping(monitor='val_accuracy', mode='max') overfitting 0.5 above
early_stopping = EarlyStopping(monitor='val_loss', mode='min') overfitting 0.5 below class
 
stop training if performance stays above or below a given threshold or baseline.
We can account for this by adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. This can be done by setting the “patience” argument.
