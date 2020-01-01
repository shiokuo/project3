from sklearn.datasets import load_digits
#from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib as mpl
import matplotlib.pyplot as plt
import random
import copy
digits = load_digits()
Xtrain,Xtest,ytrain,ytest = train_test_split(digits.data,digits.target,random_state=0)

model = RandomForestClassifier(n_estimators=64)
model.fit(Xtrain,ytrain)
ypred = model.predict(Xtest)

#print(metrics.classification_report(ypred,ytest))
#plt.show()

def test_f(func,draw=0,start=1,done=64):
    global digits
    test_lis=[]
    correctness=[]
    for test in range(start,done):
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
                    print(yp[0],i)
                    plt.show()
            
            digits.data[i]=save[:]
            #print(digits.data[i])
        test_lis.append(copy.deepcopy(test))
        correctness.append(count_error/count)
    return test_lis,correctness

#test_lis,correctness=test_f(lambda n: 0,1,done=5)
#test_lis,correctness=test_f(lambda n: int(random.randint(500,1500)*n/1000),1,start=60)
#test_lis,correctness=test_f(lambda n: n,1,done=5)
#print(dict(zip(test_lis,correctness)))

            

