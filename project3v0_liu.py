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
                digits.data[i]=func(change_ele,digits.data[i])
            
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

def test_0(n,pic):
    return pic
def test_1(n,pic):
    pic[n]=0
    return pic
def test_2(n,pic):
    pic[n]=int(random.randint(500,1500)*pic[n]/1000)
    return pic
def test_3(n,pic):
    cross=[n+1,n-1,n+8,n-8]
    for ele in cross:
        try:
            pic[ele]=pic[n]
        except:
            continue
    return pic

#test_lis,correctness=test_f(test_1,1,done=5)
#test_lis,correctness=test_f(test_2,1,start=60)
#test_lis,correctness=test_f(test_0,1,done=5)
test_lis,correctness=test_f(test_3,1,done=5)
print(dict(zip(test_lis,correctness)))

            

