#-------------------------------------------------------------------------
# AUTHOR: David Lam
# FILENAME: naive_bayes.py
# SPECIFICATION: Read the file weather_training.csv and output the classification of each of 10 instances to a file named weather_test.csv if the classification confidence is >= 0.75
# FOR: CS 4210- Assignment #2
# TIME SPENT: 4 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

#Reading the training data in a csv file
#--> add your Python code here
training_data = []
with open('weather_training.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            training_data.append(row)

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
#--> add your Python code here
outlook = {"Sunny": 1, "Overcast": 2, "Rain": 3}
temperature = {"Hot": 1, "Mild": 2, "Cool": 3}
humidity = {"High": 1, "Normal": 2}
wind = {"Strong": 1, "Weak": 2}

X = []
for data in training_data:
    X.append([outlook[data[1]], temperature[data[2]], humidity[data[3]], wind[data[4]]])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
#--> add your Python code here
class_dict = {"Yes": 1, "No": 2}
Y = [class_dict[data[5]] for data in training_data]
        
#Fitting the naive bayes to the data
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data in a csv file
#--> add your Python code here
test_data = []
with open('weather_test.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            test_data.append(row)

#Printing the header os the solution
#--> add your Python code here
print("{:<8} {:<10} {:<14} {:<12} {:<8} {:<10} {:<10}".format(
    "Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"))
print("-" * 72)  # Adjust separator line length

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
#--> add your Python code here
for data in test_data:
    test_sample = [outlook[data[1]], temperature[data[2]], humidity[data[3]], wind[data[4]]]
    probabilities = clf.predict_proba([test_sample])[0]
    
    # Determine the predicted class based on the maximum probability.
    max_prob = max(probabilities)
    
    # Only process and print if confidence is >= 0.75
    if max_prob >= 0.75:
        max_index = 0 if probabilities[0] > probabilities[1] else 1
        predicted_class = "Yes" if max_index == 0 else "No"
        confidence = "{:.2f}".format(max_prob)
        
        print("{:<8} {:<10} {:<14} {:<12} {:<8} {:<10} {:<10}".format(
            data[0], data[1], data[2], data[3], data[4], predicted_class, confidence))