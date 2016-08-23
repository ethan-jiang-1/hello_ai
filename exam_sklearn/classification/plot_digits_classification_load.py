"""
================================
Recognizing hand-written digits
================================

An example showing how the scikit-learn can be used to recognize images of
hand-written digits.

This example is commented in the
:ref:`tutorial section of the user manual <introduction>`.

"""
print(__doc__)

#import pdb; pdb.set_trace()

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pickle
import os

def get_pickle_filename():
    cwd = os.getcwd()
    filename  = os.path.normpath(cwd + "/../temp/svm.pickle")
    return filename

def save_to_file(classifier):
    s = pickle.dumps(classifier)
    filename = get_pickle_filename()

    with open(filename,"w") as f:
        f.write(s)


def load_from_file():
    filename = get_pickle_filename()

    s = None
    with open(filename,"r") as f:
        s = f.read()
    classifier_new = pickle.loads(s)
    return classifier_new

# The digits dataset
digits = datasets.load_digits()

# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 3 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# pylab.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
images_and_labels = list(zip(digits.images, digits.target))
for index, (image, label) in enumerate(images_and_labels[:4]):
    plt.subplot(2, 4, index + 1)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)

print("Training...")
# We learn the digits on the first half of the digits
classifier.fit(data[:n_samples / 2], digits.target[:n_samples / 2])


print("Save classifier...")
save_to_file(classifier)
print("load classifier...")
classifier_new = load_from_file()



# Now predict the value of the digit on the second half:
expected = digits.target[n_samples / 2:]
predicted = classifier_new.predict(data[n_samples / 2:])

print("Classification report for classifier_new %s:\n%s\n"
      % (classifier_new, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
for index, (image, prediction) in enumerate(images_and_predictions[:4]):
    plt.subplot(2, 4, index + 5)
    plt.axis('off')
    plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Prediction: %i' % prediction)

plt.show()