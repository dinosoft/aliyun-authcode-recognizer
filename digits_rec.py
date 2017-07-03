# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics

class digits_recognizer():

    def __init__(self):
        # The digits dataset
        from sklearn.datasets import fetch_mldata
        digits = fetch_mldata('MNIST original', data_home="d:/mnist/")

        # digits = datasets.load_digits()

        # The data that we are interested in is made of 8x8 images of digits, let's
        # have a look at the first 4 images, stored in the `images` attribute of the
        # dataset.  If we were working from image files, we could load them using
        # matplotlib.pyplot.imread.  Note that each image must have the same size. For these
        # images, we know which digit they represent: it is given in the 'target' of
        # the dataset.
        # images_and_labels = list(zip(digits.data, digits.target))
        #
        # for index, (image, label) in enumerate(images_and_labels[:4]):
        #     plt.subplot(2, 4, index + 1)
        #     plt.axis('off')
        #     plt.imshow(image.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
        #     plt.title('Training: %i' % label)

        # To apply a classifier on this data, we need to flatten the image, to
        # turn the data in a (samples, feature) matrix:
        n_samples = len(digits.data)
        #data = digits.data.reshape((n_samples, -1))
        data = digits.data

        # Create a classifier: a support vector classifier
        self.classifier = svm.NuSVC(nu=0.02,  kernel='rbf', gamma=0.05, cache_size=2024, shrinking=False, verbose=True)

        from sklearn.externals import joblib
        import os
        model_filename="svm_mnist.pkl"
        if os.path.isfile(model_filename):
            self.classifier = joblib.load(model_filename)
        else:
            print("training svm model...")
            # We learn the digits on the first half of the digits
            import random
            training_set = range(n_samples)
            random.shuffle(training_set)

            training_set = training_set[:n_samples/3]
            X_train_small = data[training_set]
            y_train_small = digits.target[training_set]

            ###
            # from sklearn.grid_search import GridSearchCV
            # parameters = {'nu': (0.05, 0.02), 'gamma': [3e-2, 2e-2, 1e-2]}
            #
            # svc_clf = svm.NuSVC(nu=0.1, kernel='rbf', verbose=True)
            # gs_clf = GridSearchCV(svc_clf, parameters, n_jobs=3, verbose=True)
            #
            # gs_clf.fit(X_train_small.astype('float'), y_train_small)
            #
            # print()
            # for params, mean_score, scores in gs_clf.grid_scores_:
            #     print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() * 2, params))
            # print()
            ###

            #digits.target[ digits.target == 0 ] = 10
            #x=data[training_set]
            self.classifier.fit(data[training_set]/255.0, digits.target[training_set])
            print("finished training")
            predicted = self.classifier.predict(data/255.0)

            print("finished prediction")
            from sklearn import metrics
            import numpy as np
            print("{}/{}".format( np.sum(digits.target == predicted), predicted.shape[0]) )
            # print("Classification report for classifier %s:\n%s\n"
            #       % (self.classifier, metrics.classification_report(digits.target, predicted)))
            joblib.dump(self.classifier, model_filename)

    def predict_digit(self, image):
        #prob = self.classifier.predict_proba(image.reshape(1, 64))
        ret = self.classifier.predict(image.reshape(1, 28*28))
        return ret[0]
# # Now predict the value of the digit on the second half:
# expected = digits.target[n_samples // 2:]
# predicted = classifier.predict(data[n_samples // 2:])
#
# print("Classification report for classifier %s:\n%s\n"
#       % (classifier, metrics.classification_report(expected, predicted)))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))
#
# images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
#
# plt.show()


if __name__ == "__main__":
    rec = digits_recognizer()
    # data = datasets.load_digits()
    # fea = data.images[0]
    # print "number is {}".format(rec.predict_digit(data.images[18]))
    from sklearn.datasets import fetch_mldata
    digits = fetch_mldata('MNIST original', data_home="d:/mnist/")
    n = digits.data.shape[0]
    for i in range(6):
        import random
        random_idx = random.randint(0, n)
        x = digits.data[random_idx]
        #print x.shape
        y = digits.target[random_idx]
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(x.reshape((28,28)), cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('{}, true:{}'.format(rec.predict_digit(x/255.0), y))
        # print "{}, predict {}".format(y,rec.predict_digit(x))
    plt.show()
    pass