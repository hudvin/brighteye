import os
from collections import OrderedDict

from operator import itemgetter
from sklearn import metrics, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.externals import joblib

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn.cross_validation import train_test_split, ShuffleSplit
from sklearn.svm import SVC

import pandas as pd
import numpy as np

from sklearn.learning_curve import learning_curve

import matplotlib.pyplot as plt


if __name__ == '__main__':
    work_dir = "/tmp"
    print("Loading embeddings.")
    fname = "{}/embeddings.csv".format(work_dir)
    data = pd.read_csv(fname, header=None).as_matrix()
    labels = data[:, 0]

    open("/tmp/labels.txt", "w").write("\n".join(labels.tolist()))
    # with open("/tmp/labels.csv", 'a') as f_handle:
    #     np.savetxt(f_handle, labels.tolist(), fmt="%s", delimiter=", ")


    # labels_list = list(OrderedDict.fromkeys(labels.tolist()))
    # open("/tmp/labels.txt", "w").write("\n".join(labels_list))

    embeddings = data[:,1:]

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)


    X_train, X_test, y_train, y_test = train_test_split(embeddings, labelsNum,  random_state=0)


    from sknn.platform import gpu32

    from sknn.mlp import Classifier, Layer
    from sklearn.ensemble import RandomForestClassifier
    model = nn = Classifier(
    layers=[
        Layer("Rectifier", units=200),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=10000, )

    #model = KNeighborsClassifier( n_jobs=8)

    #model = RandomForestClassifier()
    #model = svm.NuSVC(nu=0.01)

    model.fit(X_train, y_train)
    print(model)
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print "Accuracy Rate, which is calculated by accuracy_score() is: %f" % metrics.accuracy_score(expected, predicted)

    joblib.dump(model, "/tmp/model.pkl")





    print("Training for {} classes.".format(nClasses))