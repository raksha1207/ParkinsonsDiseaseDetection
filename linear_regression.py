import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def initialize_parameters(lenw):
    w=np.random.randn(1, lenw)
    #random numbers through random normal distribution
    b=0
    return w,b

def forward_prop(X,w,b):
    z=np.dot(w,X)+b
    return z
    #w has dimension 1*n
    #x has dimension n*m where n is the number of features and m is number of training samples

def cost_function(z,y):
    m=y.shape[1]
    # print(float(1.0/2*m))
    # print(np.sum(np.square(z-y)))
    J=(1.0/2.0*float(m))*np.sum(np.square(z-y))
    # print(J)
    return J

def back_prop(X,y,z):
    m=y.shape[1]
    dz=(1/m)*(z-y)
    dw=np.dot(dz,X.T)
    db=np.sum(dz)
    return dw, db

def gradient_descent_update(w,b,dw,db,learning_rate):
    w=w-learning_rate*dw
    b=b-learning_rate*db
    return w, b

def linear_regression_model(X_train,y_train,X_val,y_val,learning_rate,epochs):
    lenw=X_train.shape[0]
    w,b=initialize_parameters(lenw)
    costs_train=[]
    m_train=y_train.shape[1]
    m_val=y_val.shape[1]

    for i in range(1, epochs+1):
        z_train=forward_prop(X_train, w, b)
        #print(cost_function(z_train, y_train))
        cost_train=cost_function(z_train, y_train)
        dw, db=back_prop(X_train, y_train, z_train)
        w,b=gradient_descent_update(w,b,dw,db,learning_rate)
        if i%10==0:
            costs_train.append(cost_train)

        MAE_train=(1.0/float(m_train))*np.sum(np.abs(z_train-y_train))

        z_val=forward_prop(X_val, w, b)
        cost_val=cost_function(z_val, y_val)  
        MAE_val=(1.0/float(m_val))*np.sum(np.abs(z_val-y_val))

        print("Training Cost")
        print(cost_train)
        print("Validation Cost")
        print(MAE_train)

        print("Training MAE")
        print(cost_train)
        print("Validation MAE")
        print(MAE_val)

    plt.plot(costs_train)
    plt.xlabel("Iterations")
    plt.ylabel("Training Cost")
    plt.show

df=pd.read_csv('/Users/rajshreejain/alda/parkinsons_updrs.data')
df.head()
#print (df)

features=df.loc[:,df.columns!='total_UPDRS'].values[:,1:]
labels=df.loc[:,'total_UPDRS'].values


X=(features-features.mean()/features.max()-features.min())
X_train, X_val, y_train, y_val = train_test_split(features, labels, test_size=0.33, random_state=5)
X_train=X_train.T
y_train=np.array([y_train])
X_val=X_val.T
y_val=np.array([y_val])
linear_regression_model(X_train, y_train, X_val, y_val, 0.4, 2000)

linear_regression=linear_model.LinearRegression()
model=linear_regression.fit(X_train.T, y_train.T)
predictions=linear_regression.predict(X_val.T)

MAE_with_library=(1.0/y_val.shape[1])*np.sum(np.abs(predictions-y_val.T))

print(MAE_with_library)
