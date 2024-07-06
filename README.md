# mnist-digits-mlp-classificaition
 (MLP) neural network to classify handwritten digits from the MNIST dataset
This code implements a Multi-layer Perceptron (MLP) neural network to classify handwritten digits from the MNIST dataset. Below is an explanation of each section of the code:

1. Importing Libraries
The code starts by importing necessary libraries such as pandas, numpy, matplotlib, and classes from sklearn including MLPClassifier.

2. Loading Data
The file paths for the training and testing CSV files are defined, and the data is loaded using pandas.read_csv. The code then prints the number of columns and rows in the training dataset.

3. Separating Features and Labels
The labels (handwritten digits) are extracted from the first column of each row and stored in lists y_train and y_test. The remaining elements of each row, which are the features (pixel values of the images), are stored in lists X_train and X_test. The pixel values are normalized by dividing them by 255 to scale them between 0 and 1.

4. Scaling Features
The features are standardized using StandardScaler to ensure that they have a mean of 0 and a standard deviation of 1. This is important for training the MLP effectively.

5. Defining the MLP Classifier
An MLP classifier is defined with two hidden layers, each containing 250 neurons. The activation function is set to ReLU, and the Adam optimizer is used. The model is set to run for a maximum of 10 iterations.

6. Training the Classifier
The MLP classifier is trained on the scaled training data.

7. Making Predictions
The trained model is used to make predictions on the scaled test data.

8. Calculating Accuracy
The accuracy of the model on the test data is calculated and printed.

9. Displaying the Number of Training Labels and Predictions
The lengths of the y_train and y_pred lists are printed to verify that they match.

10. Testing with New Data
A new test input, X1_test, is provided, and the model predicts the label for this new data. The prediction is printed.
