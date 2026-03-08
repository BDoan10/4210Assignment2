#-------------------------------------------------------------------------
# AUTHOR: Brandon Doan
# FILENAME: decisionTree2.py
# SPECIFICATION: training decision trees 
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv','contact_lens_training_2.csv','contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    df = pd.read_csv(ds)

    for _, row in df.iterrows():
        dbTraining.append(row.tolist())

    #Transform categorical features to numbers and add to the 4d array X.
    # For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]

    for data in dbTraining:

        if data[0] == "Young":
            age = 1
        else:
            if data[0] == "Prepresbyopic":
                age = 2
            else:
                age = 3

        if data[1] == "Myope":
            spectacle = 1
        else:
            spectacle = 2

        if data[2] == "Yes":
            astigmatism = 1
        else:
            astigmatism = 2

        if data[3] == "Normal":
            tear = 1
        else:
            tear = 2

        X.append([age, spectacle, astigmatism, tear])

        if data[4] == "Yes":
            label = 1
        else:
            label = 2

        Y.append(label)

    accuracy_list = []

    # Loop your training and test tasks 10 times here
    for i in range(10):

        clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=5)
        clf = clf.fit(X, Y)

        correct = 0
        total = 0

        for data in dbTest:

            if data[0] == "Young":
                age = 1
            else:
                if data[0] == "Prepresbyopic":
                    age = 2
                else:
                    age = 3

            if data[1] == "Myope":
                spectacle = 1
            else:
                spectacle = 2

            if data[2] == "Yes":
                astigmatism = 1
            else:
                astigmatism = 2

            if data[3] == "Normal":
                tear = 1
            else:
                tear = 2

            class_predicted = clf.predict([[age, spectacle, astigmatism, tear]])[0]

            if data[4] == "Yes":
                true_label = 1
            else:
                true_label = 2

            if class_predicted == true_label:
                correct += 1

            total += 1

        accuracy = correct / total
        accuracy_list.append(accuracy)

    average_accuracy = sum(accuracy_list) / len(accuracy_list)

    print("Accuracy on", ds, ":", average_accuracy)
