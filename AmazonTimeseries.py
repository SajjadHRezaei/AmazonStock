# Dataset:  AMAZON Real Time stock Price. Currency in USD
# Aim: Regression

from sklearn.model_selection import cross_validate,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import plotly.graph_objects as go
import numpy as np
import pandas as pd

df=pd.read_csv('AMZN.csv')
df['Close-Open']=df.Close-df.Open
df['High-Low']=df.High-df.Low
df['DiffClose']=df['Close'].diff()
df.drop([0],axis=0,inplace=True)
df

fig = go.Figure(data=[go.Candlestick(x=df['Date'],
                open=df['Open'], high=df['High'],
                low=df['Low'], close=df['Close'])
                     ])

fig.update_layout(xaxis_rangeslider_visible=True)
fig.show()

def window(data,win_size):
    window_list=[]
    for i in range(0,len(data)-win_size+1):
        window_list.append(data[i:i+win_size])
        
    return np.array(window_list)
        

difclose=df['DiffClose'].tolist()
highlow=df['High-Low'].tolist()
closeopen=df['Close-Open'].tolist()

Data=[difclose,highlow,closeopen]
data=['difclose','highlow','closeopen']
Win_size=[5,10,30,90]
Regression_model=[SVR(),LinearRegression()]

reservior={}

for reg in Regression_model:
    d={}
    i=0
    for value in Data:
        l=[]
        for ws in Win_size:
         

            Array=window(value,ws)
            split_rate=0.9
            num_training=int(Array.shape[0]*split_rate)
            x_train=Array[:num_training,:-1]
            x_test=Array[num_training:,:-1]
            y_train=Array[:num_training,-1]
            y_test=Array[num_training:,-1]

            regressor = reg
            regressor.fit(x_train,y_train)
            score=r2_score(y_test, regressor.predict(x_test))
            #reservior[(data[i],str(ws))]=score
            l.append(score)
            print('The score for( window_size={},Regression_model={} Data={}) is : {}'.format(ws,reg,data[i],score))
        d[data[i]]=l
        i=i+1
    reservior[str(reg)]=d

df_SVR=pd.DataFrame(reservior["SVR()"])
df_SVR['Period']=Win_size
df_SVR['Model']='SVR'
df_LN=pd.DataFrame(reservior["LinearRegression()"])
df_LN['Period']=Win_size
df_LN['Model']='LinearRegression'
dfn=pd.concat([df_SVR,df_LN])
dfn.set_index('Period',inplace=True)
dfn
#-----------------------------------------------------------------------------
# Aim: Classification 

ax=df['DiffClose'].plot.hist()

# Based on histogram of DiffClose column (The dily difference of closed price ) we can consider 3 labels in the Target:
# target 0: the difference is almost constant (between -5 and 5)
# target 1: the difference is more than 5
# target -1: the difference is less than -5

df['Target']=0
df['Target'][df['DiffClose']>=5]=1
df['Target'][df['DiffClose']<=-5]=-1
df

def window(data,win_size):
    window_list=[]
    for i in range(0,len(data)-win_size+1):
        window_list.append(data[i:i+win_size])
        
    return np.array(window_list)
        

difclose=df['DiffClose'].tolist()
highlow=df['High-Low'].tolist()
closeopen=df['Close-Open'].tolist()
target=df['Target'].tolist()

Data=[difclose,highlow,closeopen]
data=['difclose','highlow','closeopen']
Win_size=[5,10,30,90]
Classification_model=[SVC(),AdaBoostClassifier(),QuadraticDiscriminantAnalysis()]

res={}

for clf in Classification_model:
    d={}
    i=0
    for value in Data:
        l=[]
        for ws in Win_size:
         

            Array=window(value,ws)
            Target=window(target,ws)
            split_rate=0.9
            num_training=int(Array.shape[0]*split_rate)
            x_train=Array[:num_training,:-1]
            x_test=Array[num_training:,:-1]
            y_train=Target[:num_training,-1]
            y_test=Target[num_training:,-1]

            classifier = clf
            classifier.fit(x_train,y_train)
            y_pred=classifier.predict(x_test)
            score=accuracy_score(y_test, y_pred)
            #reservior[(data[i],str(ws))]=score
            l.append(score)
            print('The score for( window_size={},Classification_model={} Data={}) is : {}'.format(ws,clf,data[i],score))
        d[data[i]]=l
        i=i+1
    res[str(clf)]=d
        
df_SVC=pd.DataFrame(res["SVC()"])
df_SVC['Period']=Win_size
df_SVC['Model']='SVC'

df_AD=pd.DataFrame(res["AdaBoostClassifier()"])
df_AD['Period']=Win_size
df_AD['Model']='AdaBoost'

df_Q=pd.DataFrame(res["QuadraticDiscriminantAnalysis()"])
df_Q['Period']=Win_size
df_Q['Model']='Quadratic'
dfn=pd.concat([df_SVC,df_AD,df_Q])
dfn.set_index('Period',inplace=True)
dfn

#-----------------------------------------------------------------------------

df2=df.copy()
df2['MaClose7']=df2['Close'].rolling(7).mean()
df2['MA30']=df2['Close'].rolling(30).mean()
df2['MA90']=df2['Close'].rolling(90).mean()
df2['MaCloseOpen3']=df2['Close-Open'].rolling(3).mean()
df2

n_range=0.6
n=int(df2.shape[0]*n_range)
X_train=df2.drop(['Date','Open','High','Low','Close','Adj Close','Target','Volume'],axis=1)[91:n]
Y_train=df2['Target'][92:n+1]
X_test=df2.drop(['Date','Open','High','Low','Close','Adj Close','Target','Volume'],axis=1)[n:-1]
Y_test=df2['Target'][n+1:]
print(X_train.shape[0],X_test.shape[0])
df2.drop(['Date','Open','High','Low','Close','Adj Close','Target','Volume'],axis=1)[0:92]

classifier = SVC()
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
score=accuracy_score(Y_test, y_pred)
print(score)

def num(df1,column_name):
    df=df1.copy()
    n=df.shape[0]
    l=[0]
    j=1
    for i in range(2,n+1):
        if df[column_name][i-1]*df[column_name][i]<0:
            j=1
        else:
            j=j+1
            
        if df[column_name][i-1]<0:
            l.append(-j)
        else:
            l.append(j)
            
    df['Num_'+column_name]=l
    
    return df
            
num(df,'Close-Open')

df3=num(df,'Close-Open')

n_range=0.6
n=int(df3.shape[0]*n_range)
X_train=df3.drop(['Date','Open','High','Low','Close','Adj Close','Target','Volume'],axis=1)[91:n]
Y_train=df3['Target'][92:n+1]
X_test=df3.drop(['Date','Open','High','Low','Close','Adj Close','Target','Volume'],axis=1)[n:-1]
Y_test=df3['Target'][n+1:]

classifier = SVC()
classifier.fit(X_train,Y_train)
y_pred=classifier.predict(X_test)
score=accuracy_score(Y_test, y_pred)
print(score)