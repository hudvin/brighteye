import os

from operator import itemgetter
from sklearn import metrics, svm
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier

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
    embeddings = data[:,1:]
   # embeddings = embeddings * 100000000000000000

    le = LabelEncoder().fit(labels)
    labelsNum = le.transform(labels)
    nClasses = len(le.classes_)

    nf = 256
    pca = PCA(n_components=nf)
    # X is the matrix transposed (n samples on the rows, m features on the columns)
    pca.fit(embeddings)
    X_new = pca.transform(embeddings)

    #embeddings = X_new

    X_train, X_test, y_train, y_test = train_test_split(embeddings, labelsNum, test_size=0.5, random_state=0)

    from sknn.platform import gpu32

    from sknn.mlp import Classifier, Layer
    from sklearn.ensemble import RandomForestClassifier
    model = nn = Classifier(
    layers=[
        Layer("Rectifier", units=200),
        Layer("Softmax")],
    learning_rate=0.01,
    n_iter=30000, )

    model = KNeighborsClassifier(algorithm="kd_tree", n_jobs=8)

    model = RandomForestClassifier()
   # model = svm.NuSVC(nu=0.01)


    model.fit(X_train, y_train)
    print(model)
    # make predictions
    expected = y_test
    predicted = model.predict(X_test)
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))
    print "Accuracy Rate, which is calculated by accuracy_score() is: %f" % metrics.accuracy_score(expected, predicted)

    # estimator = SVC(kernel='linear')
    # cv = ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
    # from sklearn.grid_search import GridSearchCV
    # import numpy as np
    #
    # gammas = np.logspace(-6, -1, 10)
    # classifier = GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
    # classifier.fit(X_train, y_train)
    #
    #
    # #clf.fit(embeddings, labelsNum)
    #
    #
    #
    # title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' % classifier.best_estimator_.gamma
    # estimator = SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
    # plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
    # plt.show()




    print("Training for {} classes.".format(nClasses))