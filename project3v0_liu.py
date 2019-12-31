from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import copy
digits = load_digits()
#print(digits.data)
#print(digits.target)

#fig = plt.figure(figsize=(6,6)) # figure size in inches
#fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

# plot the digits
#for i in range(64):
#    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
#    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
#    # label the image with the target value
#    ax.text(0,7,str(digits.target[i]))
# Quickly classify the digits using a random forest

Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)

model = RandomForestClassifier(n_estimators=64)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)

#print(metrics.classification_report(ypred,ytest))
#plt.show()

def test(func,draw=0,start=1):
    global digits
    for test in range(start,64):
        count=0
        count_error=0
        while count<100:
            i=random.randint(0,449)
            change_lis=[]
            while len(change_lis)<test:
                rand=random.randint(0,63)
                if not (rand in change_lis):
                    change_lis.append(rand)
                #print(change_lis)
            count+=1
            save=copy.deepcopy(digits.data[i])
            #print(save)
            for change_ele in change_lis:
                digits.data[i][change_ele]=func(digits.data[i][change_ele])
            
            yp=model.predict([digits.data[i]])
            if yp[0]!=digits.target[i]:
                count_error+=1
            if draw:
                fig = plt.figure(figsize=(6,6)) # figure size in inches
                fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
                ax = fig.add_subplot(8,8,1,xticks=[],yticks=[])
                ax.imshow(digits.images[1],cmap=plt.cm.binary,interpolation='nearest')
                # label the image with the target value
                ax.text(0,7,str(digits.target[i]))
                plt.show()
            
            digits.data[i]=save[:]
            #print(digits.data[i])
        return count_error/count,test

test(lambda n: n+1)
test(lambda n: int(random.randint(950,1050)*n/1000))
            
#fig = plt.figure(figsize=(6,6)) # figure size in inches
#fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)
## plot the digits
#for i in range(64):
#    ax = fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
#    ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
#    # label the image with the target value
#    ax.text(0,7,str(digits.target[i]))
## Quickly classify the digits using a random forest
#plt.show() 
