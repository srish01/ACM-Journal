from art.attacks.poisoning import PoisoningAttackSVM
from art.estimators.classification.scikitlearn import SklearnClassifier, ScikitlearnSVC
from art.utils import load_iris
import numpy as np 
from art.estimators.classification import SklearnClassifier
from sklearn.svm import SVC


def find_duplicates(x_train):
    dup = np.zeros(x_train.shape[0])
    for idx, x in enumerate(x_train):
        dup[idx] = np.isin(x_train[:idx], x).all(axis=1).any()
    return dup


(x_train, y_train), (x_test, y_test), min_, max_ = load_iris()
# Naturally IRIS has labels 0, 1, and 2. For binary classification use only classes 1 and 2.
no_zero = np.where(np.argmax(y_train, axis=1) != 0)
x_train = x_train[no_zero, :2][0]
y_train = y_train[no_zero]
no_zero = np.where(np.argmax(y_test, axis=1) != 0)
x_test = x_test[no_zero, :2][0]
y_test = y_test[no_zero]
labels = np.zeros((y_train.shape[0], 2))
labels[np.argmax(y_train, axis=1) == 2] = np.array([1, 0])
labels[np.argmax(y_train, axis=1) == 1] = np.array([0, 1])
y_train = labels
te_labels = np.zeros((y_test.shape[0], 2))
te_labels[np.argmax(y_test, axis=1) == 2] = np.array([1, 0])
te_labels[np.argmax(y_test, axis=1) == 1] = np.array([0, 1])
y_test = te_labels
n_sample = len(x_train)

order = np.random.permutation(n_sample)
x_train = x_train[order]
y_train = y_train[order].astype(np.float)

x_train = x_train[: int(0.9 * n_sample)]
y_train = y_train[: int(0.9 * n_sample)]
train_dups = find_duplicates(x_train)
x_train = x_train[np.logical_not(train_dups)]
y_train = y_train[np.logical_not(train_dups)]
test_dups = find_duplicates(x_test)
x_test = x_test[np.logical_not(test_dups)]
y_test = y_test[np.logical_not(test_dups)]



clean = SklearnClassifier(model=SVC(kernel='rbf'), clip_values=(-5,5))
clean.fit(x_train, y_train)

poison = SklearnClassifier(model=SVC(kernel='rbf'), clip_values=(-5,5))
poison.fit(x_train, y_train)

attack = PoisoningAttackSVM(poison, 0.01, 1.0, x_train, y_train, x_test, y_test, 25)
attack_y = np.ones((15,2)) - y_train[:15]
attack_point, _ = attack.poison(np.array(x_train[:15]), y=np.array(attack_y))

print(attack_point)
