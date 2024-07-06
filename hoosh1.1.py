import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


file_path_train = 'mnist_train.csv'
file_path_test = 'mnist_test.csv'

data_train = pd.read_csv(file_path_train)
data_test = pd.read_csv(file_path_test)

num_columns = data_train.shape[1]

print(f"تعداد ستون‌ها: {num_columns}")
num_rows = data_train.shape[0]

print(f"تعداد سطرها: {num_rows}")

column_1 = data_train.iloc[:, 0]  
y_train=list(column_1)
column_1 = data_test.iloc[:, 0]  
y_test=list(column_1)

rows_list = []
for index, row in data_train.iterrows():
    row_data = list(row)[1:]  
    for i in range(len(row_data)):
       row_data[i]= float(row_data[i])/255
    rows_list.append(row_data)
X_train = rows_list

rows_list = []
for index, row in data_test.iterrows():
    row_data = list(row)[1:]  
    for i in range(len(row_data)):
       row_data[i]= float(row_data[i])/255
    rows_list.append(row_data)
X_test = rows_list

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLP classifier
mlp = MLPClassifier(hidden_layer_sizes=(250, 250), activation='relu', solver='adam', max_iter=10)

# Train the classifier
mlp.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = mlp.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Test accuracy: {accuracy}')

print(f"y_train: {len(y_train)}")
print(f"y_pred: {len(y_pred)}")

# داده‌های ورودی برای تست
X1_test = np.array([[0,100,0,0,0,210,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,43,0,0,0,0,0,0,0,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,10,99,174,186,255,255,255,172,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,6,97,230,254,236,165,148,195,153,245,228,6,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,110,254,203,81,15,0,0,0,0,38,54,134,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,41,248,240,20,0,0,0,0,0,0,0,10,134,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,34,244,161,0,0,0,0,0,16,187,113,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,225,170,3,0,0,0,39,222,254,222,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,148,247,167,60,85,188,250,254,254,163,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,194,243,253,237,238,186,252,254,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,60,0,0,49,247,221,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,172,254,164,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,214,254,54,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,104,249,237,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,202,254,139,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,237,254,29,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,167,254,184,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,190,254,111,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,8,221,254,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,254,232,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,254,213,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,98,254,136,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
 ]])

# # پیش‌بینی برچسب‌ها برای داده‌های ورودی
predictions = mlp.predict(X1_test)

print("Predictions:", predictions)


