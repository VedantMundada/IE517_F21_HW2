import sklearn
import numpy as np
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:,:], iris.target

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=33)

scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

import matplotlib.pyplot as plt
colors = ['red', 'greenyellow', 'blue']
#it doesnt like "xrange" changed to "range"
for i in range(len(colors)):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, c=colors[i])
plt.legend(iris.target_names)
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
n=np.arange(1,91)
train_accuracy = np.empty(len(n))
test_accuracy = np.empty(len(n))
from sklearn import neighbors 
for k in range(1,90):
    knn = neighbors.KNeighborsClassifier(k ,weights='uniform')
    knn.fit(X_train, y_train)
    train_accuracy[k] = knn.score(X_train,y_train)
    test_accuracy[k] = knn.score(X_test,y_test)

import matplotlib.pyplot as plt
plt.title("K-NN : Varying number of neighbors")
plt.plot(n,test_accuracy,label="Test Accuracy")
plt.plot(n,train_accuracy,label="Train Accuracy")
plt.legend()
plt.xlabel('No of Neighbors')
plt.ylabel('Acuracy')
plt.show()

print("The best test accuracy is ",max(test_accuracy),"at k = ",np.argmax(test_accuracy)+1)

plt.scatter(train_accuracy,test_accuracy, c='blue')
plt.title("Training vs Testing accuracies")
plt.xlabel('Training Accuracies')
plt.ylabel('Testing Accuracies')
plt.show()




print("My name is Vedant Mundada")
print("My NetID is: vkm3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

