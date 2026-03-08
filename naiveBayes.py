#-------------------------------------------------------------------------
# AUTHOR: Brandon Doan
# FILENAME: naiveBayes.py
# SPECIFICATION: Classify weather test instances using Naive Bayes
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

X = []
Y = []

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
for data in dbTraining:
    if data[1] == "Sunny":
        outlook = 1
    elif data[1] == "Overcast":
        outlook = 2
    else:
        outlook = 3

    if data[2] == "Hot":
        temperature = 1
    elif data[2] == "Mild":
        temperature = 2
    else:
        temperature = 3

    if data[3] == "High":
        humidity = 1
    else:
        humidity = 2

    if data[4] == "Weak":
        wind = 1
    else:
        wind = 2

    X.append([outlook, temperature, humidity, wind])
    if data[5] == "Yes":
        Y.append(1)
    else:
        Y.append(2)

#Fitting the naive bayes to the data using smoothing
clf = GaussianNB()
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
print("Day Outlook Temperature Humidity Wind PlayTennis Confidence")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for data in dbTest:

    if data[1] == "Sunny":
        outlook = 1
    elif data[1] == "Overcast":
        outlook = 2
    else:
        outlook = 3

    if data[2] == "Hot":
        temperature = 1
    elif data[2] == "Mild":
        temperature = 2
    else:
        temperature = 3

    if data[3] == "High":
        humidity = 1
    else:
        humidity = 2

    if data[4] == "Weak":
        wind = 1
    else:
        wind = 2

    probs = clf.predict_proba([[outlook, temperature, humidity, wind]])[0]
    pred = clf.predict([[outlook, temperature, humidity, wind]])[0]
    conf= max(probs)
    if conf >= 0.75:
        if pred == 1:
            label = "Yes"
        else:
            label = "No"
        print(data[0], data[1], data[2], data[3], data[4], label, round(conf,4))
