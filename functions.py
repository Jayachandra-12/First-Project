import numpy as np
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt 
import functions as fp 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def change_into_time(df,col) :
    df[col] = pd.to_datetime(df[col])

def extract_hour(df,col) :
    df[col+'_hour'] = df[col].dt.hour

def extract_minute(df,col) :
    df[col+'_minute'] = df[col].dt.minute

def drop_column(df,col) :
    df.drop(col,axis=1,inplace=True)

def hour(x) :
    return int(x.split()[0][0:-1])

def minute(x):
    return int(x.split()[1][0:-1])

def plot(df,col) :
    fig, (ax1,ax2) = plt.subplots(2,1)
    sns.distplot(df[col],ax=ax1)
    sns.boxplot(df[col],ax=ax2)
    plt.show()

def predict(ml_model,X_train,Y_train,X_test,Y_test) :
    model = ml_model.fit(X_train,Y_train)
    print("Training score : {}".format(model.score(X_train,Y_train)))
    Y_prediction = model.predict(X_test)
    print("Predictions are : \n{}".format(Y_prediction))
    print('\n')

    r2_score = metrics.r2_score(Y_test,Y_prediction)
    print('r2_score is {}'.format(r2_score))
    print('MAE is {}'.format(metrics.mean_absolute_error(Y_test,Y_prediction)))
    print('MSE is {}'.format(metrics.mean_squared_error(Y_test,Y_prediction)))
    print('RMSE is {}'.format(np.sqrt(metrics.mean_squared_error(Y_test,Y_prediction))))
    sns.distplot(Y_test-Y_prediction)
    plt.show()

