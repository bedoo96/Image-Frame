
This work first reviews the different techniques developed over the past decades to better understand the subject, then presents our solution approach based on the use of images to represent walking, such as the Gait Energy Image (GEI ) to create the model.

We start with the conventional approach to gait recognition which includes feature extraction and classification using SVM. Feature vectors for classification were constructed using dimension reduction on cumulants calculated by principal component analysis (PCA). Then a second Deep Learning model where the representations of walking are used as input into a convolutional neural network, which is used to perform classification or to extract a feature vector which is then classified using methods of machine learning to create our second model.

The models are trained and tested with a dataset that we have built, containing seven (7) people in a 180 degree angle.
After implementation and experimentation, the results are interesting, on the other hand open to the necessary improvements. 

For Resume,This project involves first extracting frames from a video, then using the ResNet101 model for segmentation (oriented towards the COCO dataset), and storing them in a directory. Finally, it calculates the Gait Efficiency Index (GEI).

