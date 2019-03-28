#importing libraries ---------------------------------------
from sklearn.datasets import load_digits
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

#loading dataset -------------------------------------------
digits = load_digits()

X = digits.data[:-1]   #for feature dataset 
y = digits.target[:-1] #for target dataset
#print(len(X),len(y))  #for checking length of X and y

train_X,test_X,train_y,test_y = train_test_split(X,y)
#for parting dataset into training and testing dataset
#print(len(train_X),len(test_X))
#for checking length of training and testing dataset

#designing model ------------------------------------------- 
clf = svm.SVC(gamma=.0005) #designing model

clf.fit(train_X,train_y)   #fitting model

prediction = clf.predict(test_X) #target prediction for 
                                 #testing dataset
result = clf.score(test_X,test_y) #for getting accuracy of 
                                 #model over testing data

print(result*100)   #printing acuuracy
print(mean_absolute_error(test_y,prediction))
#printing mean absolute error

