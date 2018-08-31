import matplotlib.pyplot as plt
from sklearn import linear_model
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

def newsample(xTest, ytest, model):
    ar = np.array([[[1],[2],[3]], [[2],[4],[6]]])
    y = ar[1,:]
    x = ar[0,:]

    if model == 1:
        reg = linear_model.LinearRegression()
        reg.fit(x,y)
        print('least square Coefficients: \n', reg.coef_)
    if model == 2:
        reg = linear_model.Ridge (alpha = 0.1)
        reg.fit(x,y)
        print('ridged Coefficients: \n', ridge.coef_)
    if model == 3:    
        reg = linear_model.Lasso(alpha = 0.1)
        reg.fit(x,y)
        print('lasso Coefficients: \n', ridge.coef_)
        
    preds = reg.predict(xTest)

    er = []
    for i in range(len(ytest)):
        print( "actual=", ytest[i], " preds=", preds[i])
        x = (ytest[i] - preds[i]) **2
        er.append(x)

    v = np.var(er)
    print ("variance", v)
   
    print("Mean squared error (bias): %.2f" % mean_squared_error(ytest,preds))

    tst = preprocessing.scale(ytest)
    prd = preprocessing.scale(preds)
    plt.plot(tst, prd, 'g^')
    
    x1 = preprocessing.scale(xTest)
    fx = preprocessing.scale(xTest * reg.coef_)

    plt.plot(x1,fx )
    plt.show()


a = np.array([[4],[5],[6]])
b = np.array([[8.8],[14],[17]])
newsample(a,b, 1)
