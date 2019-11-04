import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split,GridSearchCV, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
print("Important libraries have been included")
# reading train data
train=pd.read_csv("Train.csv",index_col="date_time",parse_dates=True)
description=train.describe()
#%%
#
missing=train.isnull().sum()
print(missing)
#
qualitative=[f for f in train.columns if train.dtypes[f]=='object']
quantitative=[f for f in train.columns if train.dtypes[f]!='object']
#
#converting qualitative to numbers
train.is_holiday=pd.Categorical(train.is_holiday) 
train["isholiday"]=train.is_holiday.cat.codes
train.weather_type=pd.Categorical(train.weather_type) 
train["weathertype"]=train.weather_type.cat.codes
train.weather_description=pd.Categorical(train.weather_description) 
train["weatherdescription"]=train.weather_description.cat.codes
train=train.drop(["is_holiday","weather_type","weather_description"],axis=1)
#
X=train.drop(['traffic_volume'],axis=1)
Y=train['traffic_volume']
X_train,X_test,y_train,y_test=train_test_split(train[['temperature','wind_speed']],train['traffic_volume'],test_size=0.2,random_state=0)
#X_train,X_test,y_train,y_test=train_test_split(train.drop(['traffic_volume'],axis=1),train['traffic_volume'],test_size=0.2,random_state=0)
Dreg=DecisionTreeRegressor(max_depth=8,min_samples_split=10,max_leaf_nodes=100,random_state=1)
#%%
Dreg.fit(X_train,y_train)
y_pred=Dreg.predict(X_test)
mse=mean_squared_error(y_pred,y_test)
rmse=mse**0.2
print(mse," ",rmse )
#%%
#submission file
valid=pd.read_csv("Test.csv",index_col="date_time",parse_dates=True)
valid.is_holiday=pd.Categorical(valid.is_holiday) 
valid["isholiday"]=valid.is_holiday.cat.codes
valid.weather_type=pd.Categorical(valid.weather_type) 
valid["weathertype"]=valid.weather_type.cat.codes
valid.weather_description=pd.Categorical(valid.weather_description) 
valid["weatherdescription"]=valid.weather_description.cat.codes
valid=valid.drop(["is_holiday","weather_type","weather_description"],axis=1)
#%%
valid1=valid[['temperature','wind_speed']]
y_pred_valid=Dreg.predict(valid1)
valid["traffic_volume"]=y_pred_valid
sub1=valid["traffic_volume"]
sub1=sub1.to_frame()
sub1.to_csv("submission1.csv",)
#%%
# plotting variables
f = pd.melt(train, value_vars=quantitative)
g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
g = g.map(sns.distplot, "value")
#%%
for col in train.columns:
    plt.figure()
    sns.regplot(x=train[col],y=train["traffic_volume"])
#%%
train1=X_train.drop(["wind_direction","air_pollution_index","humidity","visibility_in_miles"],axis=1)
X_test1=X_test.drop(["wind_direction","air_pollution_index","humidity","visibility_in_miles"],axis=1)
Dreg1=DecisionTreeRegressor(max_depth=11,random_state=0)
X_train1=train1
Dreg1.fit(X_train1,y_train)
y_pred1=Dreg1.predict(X_test1)
mse=mean_squared_error(y_pred1,y_test)
rmse=mse**0.2
print(mse," ",rmse )
#%%
#hyperparameter tuning using GridSearchCV
model=DecisionTreeRegressor(max_depth=10,min_samples_split=5,max_leaf_nodes=100,random_state=1)
model.fit(X_train,y_train)
print("R-Squared on train dataset={}".format(model.score(X_test,y_test)))
model.fit(X_test,y_test)
print("R-Squared on test dataset={}".format(model.score(X_test,y_test)))
# hyperparameter tuning using GridSearchCV
param_grid={"criterion":["mse","mae"],
            "min_samples_split":[10,20,40],
            "max_depth":[8,10],
            "min_samples_leaf":[20,40,100],
            "max_leaf_nodes":[5,20,100]}
grid_cv_model=GridSearchCV(model,param_grid,cv=5)
grid_cv_model.fit(X_train,y_train)
print("R-Squared::{}".format(grid_cv_model.best_score_))
print("Best Hyperparameters::\n{}".format(grid_cv_model.best_params_))
#%%
# plotting the data
for c in X_train.columns:
    plt.figure()
    sns.scatterplot(x=X_train[c],y=y_train)
    print(c)
    
#%%
# removing outliers
def detect_outliers(data_1):
    threshold=3
    mean_1=data_1.mean()
    std_1=data_1.std()
    i=0
    for y in data_1:
        z_score=(y-mean_1)/std_1
        if np.abs(z_score)>threshold:
            data_1[i]=mean_1
        i=i+1

for c in X_train.columns:
    detect_outliers(X_train[c])
#%%
sns.distplot(y_train)
# checking skewness
print(y_train.skew())
target=np.log(y_train)
print(target.skew())
#%%
#separate variables into new data frames
numeric_data=train.select_dtypes(include=[np.number])
cat_data=train.select_dtypes(exclude=[np.number])
print("There are {} numeric and {} categorical columns in train data".format(numeric_data.shape[1],cat_data.shape[1]))
# for removing correlated variables plotting correlation plot
corr=numeric_data.corr()
sns.heatmap(corr)



