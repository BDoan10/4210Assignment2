#-------------------------------------------------------------------------
# AUTHOR: Brandon Doan
# FILENAME: knn.py
# SPECIFICATION: Computes the LOO-CV error rate for 1NN
# FOR: CS 4210 - Assignment #2
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')

for _, row in df.iterrows():
    db.append(row.tolist())

wrong = 0
total = 0

#Loop your data to allow each instance to be your test set
for i in range(len(db)):

    X = []
    Y = []

    # Do not forget to remove the instance that will be used for testing in this iteration.
    for j in range(len(db)):
        if j != i:
            features = [float(v) for v in db[j][0:20]]
            label = db[j][20]

            X.append(features)
            Y.append(label)

    #Test sample
    testSample = [float(v) for v in db[i][0:20]]
    correct_label = db[i][20]

    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)

    class_predicted = clf.predict([testSample])[0]

    if class_predicted != correct_label:
        wrong += 1

    total += 1

#Print the error rate
error_rate = wrong / total

print("LOO-CV Error Rate:", error_rate)
