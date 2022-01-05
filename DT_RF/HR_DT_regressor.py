import pandas as pd                                         # importing library
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("HR_DT.csv")                        # importing dataset
df

# data cleansing and eda**************************************************************************************************

df.info()                                                  # cheking for null values
df.describe()

df.duplicated().sum()                                      # checking for duplicates
df1 = df.drop_duplicates()
plt.boxplot(df.iloc[:,1:3])                                # plotting boxplot for outliers
sns.pairplot(df)                                           # plotting pair plot foreda

df1 = pd.get_dummies(df,drop_first = True)                  # getting dummy rows for categorical column

def norm1(i):                                              # scaling dataset to 0 and 1
    n = (i - i.min())/(i.max() - i.min())
    return n
norm = norm1(df1.iloc[:,[0,2,3,4,5,6,7,8,9,10]])

x = norm                                      # predictors
y = df1.iloc[:,[1]]                                              # target
# splitting dataset to train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 40)

# model building for decision tree ***************************************************************************************

regtree = tree.DecisionTreeRegressor(max_depth = 3)
regtree.fit(x_train, y_train)

# Prediction
test_pred = regtree.predict(x_test)
train_pred = regtree.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)

tree.plot_tree(regtree)
# tunnig DT ***************************************************************************************************************
regtree1 = tree.DecisionTreeRegressor(min_samples_split = 4)
regtree1.fit(x_train, y_train)

# Prediction
test_pred = regtree1.predict(x_test)
train_pred = regtree1.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)

tree.plot_tree(regtree1)
# model building by RF***************************************************************************************************

rf_clf = RandomForestRegressor(n_estimators=500, n_jobs=1, random_state=42)  # initialising and fitting model

rf_clf.fit(x_train, y_train)


# Prediction
test_pred = rf_clf.predict(x_test)
train_pred = rf_clf.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)

# RF tuninig *****************************************************************************************************************

rf_clf_grid =  RandomForestRegressor( n_estimators = 500 ,n_jobs=1, random_state=50)  # initialising and fitting rf

# parameters for grid search
param_grid = {"max_features": [2, 3, 5, 7, 8], "min_samples_split": [2, 3, 7],'n_estimators': [10,15,20,24,25,30,50]}

# initialising grid search
grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)                             # fitting grid serach

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_                 # best estimator

# Prediction
test_pred =cv_rf_clf_grid .predict(x_test)
train_pred =cv_rf_clf_grid.predict(x_train)

# Error on test dataset
mean_squared_error(y_test, test_pred)
r2_score(y_test, test_pred)

# Error on train dataset
mean_squared_error(y_train, train_pred)
r2_score(y_train, train_pred)
# plotting best tree
tree.plot_tree(cv_rf_clf_grid.estimators_[9], feature_names = x.columns,class_names=['0', "1"],filled=True)
 
# checking and plotting best features 

imp_df = pd.DataFrame({
    "Varname": x_train.columns,
    "Imp": cv_rf_clf_grid.feature_importances_
})

imp_df.sort_values(by="Imp", ascending=False)
 
plt.plot(imp_df.Varname ,imp_df.Imp , "ro-");plt.xticks(rotation = 90)

