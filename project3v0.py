from sklearn.datasets import load_digits
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
digits = load_digits()
digits.keys()
i=1
while i <= len(digits):
    for k in range(0,64):
        digits.data[i,k] = int(digits.data[i,k]*4/5+random.random*3)
    i+=1
fig = plt.figure(figsize=(6,6)) # figure size in inches
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
    # label the image with the target value
    ax.text(0,7,str(digits.target[i]))
# Quickly classify the digits using a random forest
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)

model = RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)

print(metrics.classification_report(ypred,ytest))

plt.show()