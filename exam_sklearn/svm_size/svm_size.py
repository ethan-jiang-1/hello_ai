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

def get_temp_dir():
    cwd = os.getcwd()
    paths = cwd.split('/')
    root_paths = []
    for path in paths:
        if path == 'hello_ai':
            root_paths.append(path)
            break
        root_paths.append(path)
    dir_tmp = '/'.join(root_paths) + "/temp/svm_footprint"

    if not os.path.isdir(dir_tmp):
        os.mkdir(dir_tmp)

    return dir_tmp

def get_pickle_filename(index):
    #import pdb; pdb.set_trace()
    temp_dir = get_temp_dir()
    filename  = os.path.normpath(temp_dir + "/svm.{0:03}.tmp".format(index))
    return filename

def save_to_file(classifier, index):
    s = pickle.dumps(classifier)
    filename = get_pickle_filename(index)

    with open(filename,"w") as f:
        f.write(s)


def load_from_file(index):
    filename = get_pickle_filename(index)

    s = None
    with open(filename,"r") as f:
        s = f.read()
    classifier_new = pickle.loads(s)
    return classifier_new



def train(data,digits,num_traning_sample):
    tindex = 10
    classifier_new = None
    while tindex < num_traning_sample:

        #import pdb; pdb.set_trace()
        classifier = svm.SVC(gamma=0.001)

        classifier.fit(data[:tindex], digits.target[:tindex])

        print("Save classifier " + str(tindex))
        save_to_file(classifier,tindex)
        print("load classifier " + str(tindex))
        classifier_new = load_from_file(tindex)

        tindex += 10

        if tindex %20 == 0:
            predict_result(data,digits,classifier_new,tindex)

    return classifier_new


def predict_result(data,digits,classifier_new, num_traning_sample):
    n_samples = len(digits.images)

    # Now predict the value of the digit on the second half:
    expected = digits.target[n_samples / 2:]
    predicted = classifier_new.predict(data[n_samples / 2:])

    print("Number of traning samples: " + str(num_traning_sample))
    print("Classification report for classifier_new %s:\n%s\n" % (classifier_new, metrics.classification_report(expected, predicted)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

    return predicted



def main():

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

    #training
    #num_traning_sample = n_samples
    num_traning_sample = 400
    classifier_new = train(data,digits,num_traning_sample)

    # Now predict the value of the digit on the second half:
    predicted = predict_result(data,digits,classifier_new,num_traning_sample)


    images_and_predictions = list(zip(digits.images[n_samples / 2:], predicted))
    for index, (image, prediction) in enumerate(images_and_predictions[:4]):
        plt.subplot(2, 4, index + 5)
        plt.axis('off')
        plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.title('Prediction: %i' % prediction)

    plt.show()



main()