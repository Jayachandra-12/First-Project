import numpy as np
import pandas as pd  
import seaborn as sns 
import matplotlib.pyplot as plt 
import functions as fp 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split



train_data = pd.read_excel("C:\Program Files\Python39/Data_Train.xlsx")
#print(train_data.head())
#print(train_data.isna().sum())
train_data.dropna(inplace=True)
#print(train_data.isna().sum())
#print(train_data.columns)
#changing object to datetime format
for i in ['Date_of_Journey', 'Dep_Time', 'Arrival_Time'] :
    fp.change_into_time(train_data,i)
#print(train_data.dtypes)

train_data['journey_date'] = train_data['Date_of_Journey'].dt.day
train_data['journey_month'] = train_data['Date_of_Journey'].dt.month
fp.drop_column(train_data,'Date_of_Journey')


fp.extract_hour(train_data,'Dep_Time')
fp.extract_minute(train_data,'Dep_Time')
fp.drop_column(train_data,'Dep_Time')


fp.extract_hour(train_data,'Arrival_Time')
fp.extract_minute(train_data,'Arrival_Time')
fp.drop_column(train_data,'Arrival_Time')

#duration time
duration = list(train_data['Duration'])

for i in range(len(duration)) :
    if len(duration[i].split()) == 2 :
        pass
    else :
        if 'h' in duration[i] :
            duration[i] = duration[i] + ' 0m'
        else :
            duration[i] = '0h ' + duration[i]
train_data['Duration'] = duration

train_data['Duration_hours'] = train_data['Duration'].apply(fp.hour)
train_data['Duration_minutes'] = train_data['Duration'].apply(fp.minute)
fp.drop_column(train_data,'Duration')


cat_col = []
for i in train_data.columns :
    if train_data[i].dtypes == 'object' :
        cat_col.append(i)
cont_col = []
for i in train_data.columns :
    if train_data[i].dtypes != 'object' :
        cont_col.append(i)
#print(cat_col)


#print(train_data.head())
#print(train_data.dtypes)

#dealing with categorical data

categorical = train_data[cat_col]

#print(categorical['Airline'].value_counts())
#print(categorical.head())

Airline = pd.get_dummies(categorical['Airline'], drop_first=True)
#print(Airline.head())

Source = pd.get_dummies(categorical['Source'],drop_first=True)
#print(Source.head())

Destination = pd.get_dummies(categorical['Destination'], drop_first=True)
#print(Destination.head())

#Now we are comsidering route
categorical['Route_1'] = categorical['Route'].str.split('→').str[0]
categorical['Route_2'] = categorical['Route'].str.split('→').str[1]
categorical['Route_3'] = categorical['Route'].str.split('→').str[2]
categorical['Route_4'] = categorical['Route'].str.split('→').str[3]
categorical['Route_5'] = categorical['Route'].str.split('→').str[4]
fp.drop_column(categorical,'Route')
#print(categorical.isnull().sum())
#print(categorical.head())
for i in ['Route_3','Route_4','Route_5'] :
    categorical[i].fillna('None',inplace=True)
#print(categorical.isnull().sum())
#print(categorical.head())

encoder = LabelEncoder()

for i in ['Route_1','Route_2','Route_3','Route_4','Route_5'] :
    categorical[i] = encoder.fit_transform(categorical[i])
#print(categorical.head())

#Additional info is same for every value so we drop it
fp.drop_column(categorical,'Additional_Info')
#print(categorical.head())

#Now we need to change number of stops into numbers
#print(categorical['Total_Stops'].unique())
dict = {'non-stop':0, '2 stops':2, '1 stop':1, '3 stops':3, '4 stops':4}
categorical['Total_Stops'] = categorical['Total_Stops'].map(dict)
#print(categorical.head())

#print(categorical.head())
data_train = pd.concat([train_data[cont_col],categorical,Airline,Source,Destination],axis=1)
fp.drop_column(data_train,'Airline')
fp.drop_column(data_train,'Source')
fp.drop_column(data_train,'Destination')


#print(data_train.head())
#print(data_train.columns)
#print(data_train.dtypes)

#Until here excel sheet is completely turned into numerical data

#fp.plot(data_train,'Price')
#checking and changing outliers

data_train['Price'] = np.where(data_train['Price'] >= 40000,data_train['Price'].median(),data_train['Price'])
#fp.plot(data_train, 'Price')

#seperating independent and dependent variable

X = data_train.drop('Price',axis=1)
#print(X.head())
Y = data_train['Price']
#print(Y.head())

#print(X.shape)

#giving priority to every column

imp = pd.DataFrame(mutual_info_classif(X,Y),index=X.columns)
imp.columns = ['importance']
#print(imp)
imp.sort_values(by='importance',ascending=False,inplace=True)
#print(imp)

#Now we are performing predictions

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
fp.predict(RandomForestRegressor(),X_train,Y_train,X_test,Y_test)

