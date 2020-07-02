import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import csv

# read the csv file
train_data= pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')

#use fillna to fill empty cells and .median() to replace them with the median value of the column
train_data['Age'].fillna((train_data['Age'].median()), inplace=True)
train_data['Year of Record'].fillna((train_data['Year of Record'].median()), inplace=True)

# use ffill to fill empty cells
train_data['Profession'].fillna(method='ffill', inplace=True)
train_data['Gender'].fillna(method='ffill', inplace=True)
train_data['University Degree'].fillna(method='ffill', inplace=True)
train_data['Hair Color'].fillna(method='ffill', inplace=True)
#train_data=train_data.fillna(method='ffill')
test_data= pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')

test_data['Age'].fillna((test_data['Age'].median()), inplace=True)
test_data['Year of Record'].fillna((test_data['Year of Record'].median()), inplace=True)


test_data['Profession'].fillna(method='ffill', inplace=True)
test_data['Gender'].fillna(method='ffill', inplace=True)
test_data['University Degree'].fillna(method='ffill', inplace=True)
test_data['Hair Color'].fillna(method='ffill', inplace=True)
test_data.isnull().sum()/len(test_data)

#Splitting the data for training and validation
len_train = len(train_data)
len_test_split=int(len_train/10)
len_train_split = len_train-len_test_split


test_split_data=train_data.iloc[len_train_split:,:]
train_split_data=train_data.iloc[:len_train_split,:]
pred_data=test_data

combineData = pd.concat([train_split_data, test_split_data,pred_data])



combineData_X = combineData[['Country','Age','Profession','University Degree','Gender','Hair Color','Year of Record','Body Height [cm]','Size of City']]

combineData_Y = combineData[['Income in EUR']]

#label encoding and one hot encoding the data
le=LabelEncoder()
combineData_X['Country']=le.fit_transform(combineData_X['Country'])
combineData_X['Profession']=le.fit_transform(combineData_X['Profession'])
combineData_X['University Degree']=le.fit_transform(combineData_X['University Degree'])
combineData_X['Gender']=le.fit_transform(combineData_X['Gender'])
combineData_X['Hair Color']=le.fit_transform(combineData_X['Hair Color'])



EncodeFmt=OneHotEncoder(categorical_features=[5])
combineData_X=EncodeFmt.fit_transform(combineData_X).toarray()

training_data_X=combineData_X[:len_train_split,:]

training_data_Y=combineData_Y.iloc[:len_train_split,:]


from sklearn.ensemble import RandomForestRegressor

#applying Random Forest Regressor
model=RandomForestRegressor()

# model=LinearRegression()
#model=DecisionTreeRegressor() 

result=model.fit(training_data_X,training_data_Y)

test_split_x=combineData_X[len_train_split:len_train,:]
test_split_y=combineData_Y.iloc[len_train_split:len_train,:]

predict_data=combineData_X[len_train:,:]

from sklearn.metrics import mean_squared_error
from math import sqrt
y_predicted=model.predict(test_split_x)
y_actual=test_split_y
rms = sqrt(mean_squared_error(y_actual, y_predicted))

print(rms)

result=model.predict(predict_data)

df = pd.DataFrame(result)
df.to_csv("output.csv", header=['Income'])
print(model.score(test_split_x,test_split_y))

