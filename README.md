
# pneumonia_detection_analysis
* Hardware or software configuration

   For the model training we used google colab-GPU to run the model and process the charts. We encountered few issues while running model on local machine such as shutting down    of kernel due to delayed times.

* Did you experience any memory or other difficulties with the image or data size?
   
  Severe Overfitting
 very long training time, installing new packages such as cvs and updating version of tensorflow.

* What does the model do if it receives an image that is not an x-ray (for example, a dog)?

We are working on a model that will put all other images in “Other” category and it is still in testing phase  but due to time constrain of the project , it is not included as fpr now.

* Did you consider or explore other CNN model architectures and other layers?

 We tried various hyperparameters such as different activation functions, adding dropout layers, increasing/decreasing nodes .to work with the model
 
* Can you display graphs for the metrics over the training epochs (train loss, test loss, etc.)? What were the final numbers?



* How long did the training take?

The training took about 15 min for 50 epochs on local machine and google colab-GPU took about 10 min.


* Note that I deducted a few minor points from the ML grade due to the lack of information on other model hyperparameters that you may have tried and how you ended up with this exact model design.
We tried various hyperparameters such as different activation functions such as "adam", "", adding dropout layers, increasing/decreasing nodes .to work with the model

The early stopping criteria would also help avoid overfitting

RMSProp, which stands for Root Mean Square Propagation, is a gradient descent optimization algorithm.
made sure not to use drop out layer in earlier layers to not drop imoprtant faetures
Adam, derived from Adaptive Moment Estimation, is an optimization algorithm. The Adam optimizer makes use of a combination of ideas from other optimizers.
Adam realizes the benefits of both AdaGrad and RMSProp
 Early stopping is a method that allows you to specify an arbitrary large number of training epochs and stop training once the model performance stops improving on a hold out validation dataset.
 
 Often, the first sign of no further improvement may not be the best time to stop training. This is because the model may coast into a plateau of no improvement or even get slightly worse before getting much better.

We can account for this by adding a delay to the trigger in terms of the number of epochs on which we would like to see no improvement. This can be done by setting the “patience” argument.

early_stopping = EarlyStopping(monitor='val_accuracy', mode='max') overfitting 0.5 above
 early_stopping = EarlyStopping(monitor='val_loss', mode='min') overfitting 0.5 below class
 
 baseline increase normal
 decrease pneumonia
 
 patience increase pneumonia
 decrease normal
