import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics

from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import cv2
import numpy as np
import os

class knn_clf():

    def __init__(self):
        # The digits dataset

        #data = digits.data.reshape((n_samples, -1))

        # Create a classifier: a support vector classifier

        model_filename="knn_model.pkl"
        if os.path.isfile(model_filename):
            self.classifier = joblib.load(model_filename)
        else:
            self.classifier = KNeighborsClassifier(n_neighbors=1)
            import glob
            data = np.zeros((10, 20*30))
            target = np.zeros((10, 1))
            for i in range(10):
                f = "D:/opencv_camera/digits/{}.png".format(i)
            # for f in glob.glob("D:/camera_digit/*.jpg"):
                img = cv2.imread(f)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (20, 30), interpolation=cv2.INTER_CUBIC)
                data[i, :]=255.0-img.reshape((-1))
                target[i]=i

            self.classifier.fit(data, target)
            joblib.dump(self.classifier, model_filename)

    def predict_digit(self, image):
        #prob = self.classifier.predict_proba(image.reshape(1, 64))
        image = image/1.0
        ret = self.classifier.predict(image.reshape(1, 20*30))
        return ret[0]

if __name__ == "__main__":
    rec = knn_clf()
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
        x = x.reshape(28, 28)
        x = cv2.resize(x, (20, 30), interpolation=cv2.INTER_CUBIC)
        #print x.shape
        y = digits.target[random_idx]
        plt.subplot(2, 4, i + 1)
        plt.axis('off')
        plt.imshow(x, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('{}, true:{}'.format(rec.predict_digit(x), y))
        plt.show()
    pass
