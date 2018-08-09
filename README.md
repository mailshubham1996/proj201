# proj201
This is a basic project on application of principles of machine learning with cv2.The aim of this project is to classify whether the leaf is diseased or healthy or likely between the two categories.
To maintain the tradeoff between precision and recall we?I have decided on a threshold of svm score 15
This is to ensure to reduce risk and correct action as  the lct is basically a recall oriented task as the risks of false positves are greater than that of false negatives and hence human filters can be applied to check for false negative cases given their coordinates.


Following are the sequence of steps followed in this:
1> obtain the set of images . rename the unhealthy ones between 0 and 1000 and healthy ones between 1000 and above.resize them to 255*255

2> use contrast enhancement to increase quality.Then use foreground extraction techniques to obtain the desired area.


3>Then we have 2 methods.


Method 1
1>use of bag of words module.for each image use sift/surf feature extraction to extract feature points.Add them to your training dataset .

2>then create trainlabels[] and traindata[] using the above BOW module.trainlabels set to 0 for diseased and 1 for healthy.Then for each image .compute() method will give the trainning array.add this array to the traindata.

3>Then for the test image calculate the feature and then use the .compute method to get the word(array).use predict of svm for the required prediction



Method 2

1>extract the features . use pca method for reduction.Then flatten it and keep adding them.


2>


