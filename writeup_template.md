# **Traffic Sign Recognition** 

by Yaser Ahmed, Georgia Institue of Technology (2018)

## Abstract
The goal of this project is to use deep learning to classify german traffic signs, with validation set accuracy of 93% or better. In particular, I implement and train the LeNet architecture using the tensorflow framework. The original LeNet architecture is modified to take 32x32 grayscale images, and output 43 different traffic sign classes. I also demonstrate data augmentation techniques in the domain of image recognition, which helps improve the accuracy of the model. I also retrieve the final softmax probabilities of the test examples, which provide additional insight into what the model has learned.

### Pipeline
The project implements the following pipeline:

* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./training_validation_histogram.png "Visualization"
[image2]: ./example_grayscale.png "Grayscaling"
[image3]: ./augmented_image.png "Shift Left"
[image4]: ./learning_curve_X_train_norm_20epochs.png "Early Learning Curve"
[image5]: ./learning_curve_X_train_norm_20epochs_dropout.png "Final Learning Curve"
[image6]: ./1.jpg "Traffic Sign 1"
[image7]: ./9.jpg "Traffic Sign 2"
[image8]: ./14.jpg "Traffic Sign 3"
[image9]: ./17.jpg "Traffic Sign 4"
[image10]: ./22.jpg "Traffic Sign 5"

### Data Set Summary & Exploration

I used the python pandas library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32
* The number of unique classes/labels in the data set is 43

The following is a histogram showing a distribution of images with each label in the training set.

![alt text][image1]

### Data Pre-processing

As a first step, I convert the images to grayscale. This could take care of certain distortions, e.g. shadows, different quality of colors from different sources, etc. To be sure, we do lose some information because color in traffic signs does convey important meaning, but as shown later, in this case the desired accuracy was achieved with grayscale. As a future enhancement all channels can be considered. It will impact performance due to additional channels and may also require a larger network.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

After that, I normalized the image data. In machine learning it is generally a good practice to normalize data for many algorithms. Some desirable effects of doing this are:
1. It brings all features (pixels in this case) to the same scale so some features don't overshadow others. For example, in traffic sign images several pixels that represent empty white areas will end up affecting the model disproportionately and prevent the algorithm from learning important features that have low pixel values.
1. It brings different samples to the same scale to undo effects of unimportant features, e.g., different brightness levels.
1. It ensures numerical stability.

I also decided to augment the data because some of the signs had very few examples. To add more data to the the data set, I identified all labels with less than 500 examples in the training set, and shifted those images left and right by a single pixel.

Here is an example of an original image and an augmented image shifted left by 1 pixel:

![alt text][image3]

### Model Architecture and Selection Approach

My final model consisted of the following layers:

| Layer         		      |     Description	        			                 		| 
|:---------------------:|:---------------------------------------------:| 
| Input         		      | 32x32x1 RGB image   							                   | 
| Convolution 5x5x6     | 1x1 stride, no padding, outputs 28x28x6      	|
| RELU					             |										                                   		|
| Max pooling	      	   | 2x2 stride,  outputs 14x14x6  	            			|
| Convolution 5x5x1     | 1x1 stride, no padding, outputs 10x10x16   			|
| RELU					             |										                                   		|
| Max pooling	      	   | 2x2 stride,  outputs 5x5x16   	            			|
| Fully connected		     | Input=400, output 120		                   				|
| RELU                  |                                               |
| Fully connected       | output 84                                     |
| RELU                  |                                               |
| Fully connected       | output 43                                     |
|					                 	|						                                   						|

For training, I used the AdamOptimizer, batch size of 128, 20 epochs, and learning rate of 0.001

My final model results were:
* Training set accuracy: 0.877
* Validation set accuracy: 0.937
* Test set accuracy: 0.918

* I started off with the LeNet architecture, since it is known to perform well on the MNIST image data set, and the traffic sign dataset also consists of similar type of images. Its architecture is also simple enough as far as CNNs go, so that its sufficiently fast to run multiple iterations on a personal laptop with only dual or quadcores without the use of a GPU. I started off with 10 epochs and increased to 20 epochs. The validation set accuracy remained around 0.85 while the training set accuracy was high between 0.98 to 0.99, indicating overfitting. See learning curve below for early runs:

![alt text][image4]

To take care of that, I added a dropout after each activation. I started off with a training dropout rate of 0.50. It improved the validation set accuracy to around 0.91, but the trining set accuracy was lower around 0.85, which raised concern for underfitting. So I reduced the dropout rate to 0.80. This brought the validation set accuracy above the required 0.93, and ultimately the test set accuracy of 0.91, eventhough the training set accuracy was around 0.87. Here is learning curve for the final run:

![alt_text][image5]

I stopped here since the target accuracy was achieved. Further possible improvements include increasing the number of epochs, augmenting the data set further, playing with the filter size, or increasing the breadth or depth of the architecture.

### Test the Model on New Images from the Web

Here are five German traffic signs that I found on the web:

![alt text][image6] 
![alt text][image7] 
![alt text][image8] 
![alt text][image9]
![alt text][image10]

The first image might be difficult to classify because the background maybe different from the training set. For this one particular, the (presumably) dark red garage door in the bottom right kind of meshes with the red cicle of the sign. However the network seem to identify this with pretty high accuracy.

The second image might be difficult to classify because again the background differences from the training set. In this particular case, 1/3rd of the image background is white, and bright orange on the bottom right may cause the network to misidentify the boundary.

The third image maybe difficult because the image is taken from an angle and appears somewhat skewed, which maybe the case with a real camera or video frame capturing the image.

The fourth image of no-entry sign maybe difficult because there are writings scratched in and stickers pasted on the sign, which again maybe the case with signs encountered in the real world.

The last image maybe difficult because the image is somewhat faded.

Here are the results of the prediction:

| Image			              |     Prediction	                          					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h             		| 30 km/h     				                         					| 
| No Passing 	        		| No Passing		                           							|
| Stop   	            		| Stop                                   							|
| No Entry 	          		| No Entry  				                          	 				|
| Bumpy Road   			      | Bumpy Road                             							|


The model was able to correctly guess all of the traffic signs, which gives an accuracy of 100%. The test set accuracy was 91.8%. The web images were not necessarily chosen to be challenging to classify. The test set has a lot more images with presumably more variety so it is understandable that the accuracy of test set is lower, and it is a better sample of traffic signs than the 5 web images.

#### Exploring the Softmax probabilities for each prediction

For most images, the model is more than 99% sure of their correct labels.

For the first image, 30 km/h sign, the top five soft max probabilities were:

| Probability          	|     Prediction	                          					| 
|:---------------------:|:---------------------------------------------:| 
| .999        		       	| 30 km/h   						                           			| 
| 9.9e-05     				      | 20 km/h					                             					|
| 1.5e-07				           | 50 km/h							                             			|
| 9.9e-08      		      	| 70 km/h				 	                              			|
| 2.3e-09 	             | 80 km/h                                							|

The top five soft max probabilties for No Passing sign were:

| Probability          	|     Prediction	                          					| 
|:---------------------:|:---------------------------------------------:| 
| .998        		       	| No Passing						                           			| 
| 7.4e-04     				      | End of No Passing				                  					 	|
| 5.8e-04				           | Vechiles over 3.5 metric tons prohibited      |
| 2.1e-04      		      	| Priority Road				 		                        		|
| 2.7e-05		             | No Passing for vehicle over 3.5 metric tons   |

Top five soft max probabilities for Stop sign were:

| Probability          	|     Prediction	                          					| 
|:---------------------:|:---------------------------------------------:| 
| .996          		      | Stop				                           			| 
| 7.6e-04    				       | Keep right			                  					 	|
| 4.8e-04				           | 50 km/h    |
| 3.8e-04      		      	| Yield			 		                        		|
| 2.4e-04		             | Road Work   |

Top five soft max probabilities for No Entry:

| Probability          	|     Prediction	                          					| 
|:---------------------:|:---------------------------------------------:| 
| .997          		      | No Entry				                           			| 
| 1.1e-03     				      | Stop		                  					 	|
| 7.3e-04               | Turn Left Ahead  |
| 3e-04        		      	| End of all speed and passing limits         		|
| 1.4e-04 		            | Turn right ahead    |


Top five soft max probabilities for Bumpy Road:

| Probability          	|     Prediction	                          					| 
|:---------------------:|:---------------------------------------------:| 
| .9999         		      | Bumpy Road  				                           			| 
| 2.7e-06   				        | Bicycle Crossing	                  					 	|
| 1.9e-08				           | Road Work  |
| 2.5e-09      		      	| Turn Leaft Ahead 		                        		|
| 1.5e-09 		            | Dangerous curve to the right    |


