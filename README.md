
# pneumonia_detection_analysis
* For the training - are you using a GPU or a CPU? Local machine or online 
resource? Any other details on the hardware or software configuration?

For the model training we used google colab-GPU and local computer

* Did you experience any memory of other difficulties with the image or data size?
We are working with over 5200 images
yes very long training time, installing new packages such as cvs and updating version of tensorflow.

* What does the model do if it receives an image that is not an x-ray (for example, a dog)?

We are working on a model that will put all other images in “Other” category and it is still in testing phase  but due to time constrain of the project , it is not included as such.

* Did you consider or explore other CNN model architectures and other layers?

yes

* Can you display graphs for the metrics over the training epochs (train loss, test loss, etc.)? What were the final numbers?

* How long did the training take?

The training took about an hour for 20 epochs on local machine and google colab-GPU took about 25 min
For 50 epochs


* Note that I deducted a few minor points from the ML grade due to the lack of information on other model hyperparameters that you may have tried and how you ended up with this exact model design.
