#-------------------------------------------------------------------------
# AUTHOR: David Lam
# FILENAME: knn.py
# SPECIFICATION: read the file email_classification.csv and compute LOO-CV error rate for a 1NN classifier on the spam/ham classification task
# FOR: CS 4210- Assignment #2
# TIME SPENT: 3 hours
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard vectors and arrays

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import csv

db = []

#Reading the data in a csv file
with open('email_classification.csv', 'r') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0: #skipping the header
            db.append (row)

error_count = 0  # Initialize error count to calculate misclassifications

#Loop your data to allow each instance to be your test set
for i in db:
    
    #Add the training features to the 2D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 3], [2, 1,], ...]].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here
    X = []
    for row in db:
        if row != i:
            # Convert all feature values (all columns except the last) to float and add to X
            features = [float(value) for value in row[:-1]]
            X.append(features)

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    #--> add your Python code here 
    classes = {'spam': 1, 'ham': 2}
    Y = [float(classes[row[-1]]) for row in db if row != i]

    #Store the test sample of this iteration in the vector testSample
    #--> add your Python code here
    testSample = [float(value) for value in i[:-1]]

    #Fitting the knn to the data
    clf = KNeighborsClassifier(n_neighbors=1, p=2)
    clf = clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2]])[0]
    #--> add your Python code here
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    #--> add your Python code here
    if class_predicted != float(classes[i[-1]]):
        error_count += 1

#Print the error rate
#--> add your Python code here
error_rate = error_count / len(db)
print(f"LOO-CV error rate = {error_rate:.2f}")






