import sklearn
print( 'The scikit learn version is {}.'.format(sklearn.__version__))
from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:,:], iris.target

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.35, random_state=33)

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
plt.title("Iris Data Set")
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()

from sklearn.tree import DecisionTreeClassifier, plot_tree
dtc = DecisionTreeClassifier(max_depth=4)
dtc.fit(X_train, y_train)
y_pred = dtc.predict(X_test)
from sklearn import metrics
accuracy= (metrics.accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plot_tree(dtc, filled=True)
plt.show()
plt.show()

print("The test accuracy is ",accuracy)



from sklearn import metrics
y_train_pred = dtc.predict(X_train)
print( "The training accuracy is ",metrics.accuracy_score(y_train, y_train_pred) )


y_pred = dtc.predict(X_test)


print( metrics.classification_report(y_test, y_pred, target_names=iris.target_names) )


print( metrics.confusion_matrix(y_test, y_pred) )


print("My name is Vedant Mundada")
print("My NetID is: vkm3")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

