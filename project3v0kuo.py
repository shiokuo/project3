from sklearn.datasets import load_digits
import random
digits = load_digits()
digits.keys()
import matplotlib as mpl
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6,6)) # figure size in inches
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

# plot the digits
#for i in range(64):
#    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
#    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    # label the image with the target value
 #   ax.text(0,7,str(digits.target[i]))

# Quickly classify the digits using a random forest
from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)
#ypred = model.predict(Xtest)
from sklearn import metrics
#print(metrics.classification_report(ypred,ytest))
#print(digits.data)



for i in digits.data:
    n=0
    while n<len(i)-1:
        i[n]=i[n+1]
        n+=1
print(digits.data)
for i in range(64):
    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    # label the image with the target value
    ax.text(0,7,str(digits.target[i]))
plt.show()
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)
ypred = model.predict(Xtest)
print(metrics.classification_report(ypred,ytest))