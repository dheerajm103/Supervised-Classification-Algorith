import pandas as pd                                         # importing library
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier as DT
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("Fraud_check.csv")                        # importing dataset
df

# data cleansing and eda**************************************************************************************************

df.info()                                                  # cheking for null values
df.describe()

df.duplicated().sum()                                      # checking for duplicates

df['Taxable.Income'] = np.where(df['Taxable.Income'] <= 30000, 0, df['Taxable.Income'])   #  converting sales to 0 and 1
df['Taxable.Income'] = np.where(df['Taxable.Income'] > 30000, 1, df['Taxable.Income'])

plt.boxplot(df.iloc[:,[3,4]])                                # plotting boxplot for outliers
sns.pairplot(df)                                           # plotting pair plot foreda

df = pd.get_dummies(df,drop_first = True)                  # getting dummy rows for categorical column

def norm1(i):                                              # scaling dataset to 0 and 1
    n = (i - i.min())/(i.max() - i.min())
    return n
norm = norm1(df)

x = norm.iloc[:,1:7]                                       # predictors
y = norm.iloc[:,[0]]                                              # target
# splitting dataset to train and test

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 40)

# model building for decision tree ***************************************************************************************

model = DT(criterion = 'entropy')                        # initialising decision tree 
model.fit(x_train,y_train)                               # fitting model
pred1 = model.predict(x_test)

confusion_matrix(y_test, model.predict(x_test))          # checking accuracy
accuracy_score(y_test, model.predict(x_test))
accuracy_score(y_train, model.predict(x_train))

tree.plot_tree(model)                                   # plotting tree

# tunnig DT ***************************************************************************************************************

model1 = DT(criterion = 'entropy' , max_depth = 6)      # initialising and fitting model
model1.fit(x_train,y_train)
pred2 = model1.predict(x_test)

confusion_matrix(y_test, model1.predict(x_test))          # checking accuracy
accuracy_score(y_test, model1.predict(x_test))
accuracy_score(y_train, model1.predict(x_train))

tree.plot_tree(model1)                                   # plotting tree

# model building by RF***************************************************************************************************

rf_clf = RandomForestClassifier(n_estimators=500, n_jobs=1, random_state=42)  # initialising and fitting model

rf_clf.fit(x_train, y_train)


confusion_matrix(y_test, rf_clf.predict(x_test))            # cheking accuracy
accuracy_score(y_test, rf_clf.predict(x_test))
accuracy_score(y_train, rf_clf.predict(x_train))

# RF tuninig *****************************************************************************************************************

rf_clf_grid = RandomForestClassifier( n_estimators = 500 ,n_jobs=1, random_state=50)  # initialising and fitting rf

# parameters for grid search
param_grid = {"max_features": [2, 3, 5, 7, 8], "min_samples_split": [2, 3, 7],'n_estimators': [10,15,20,24,25,30,50]}

# initialising grid search
grid_search = GridSearchCV(rf_clf_grid, param_grid, n_jobs = -1, cv = 5, scoring = 'accuracy')

grid_search.fit(x_train, y_train)                             # fitting grid serach

grid_search.best_params_

cv_rf_clf_grid = grid_search.best_estimator_                 # best estimator

confusion_matrix(y_test, cv_rf_clf_grid.predict(x_test))     # checking accuracy
accuracy_score(y_test, cv_rf_clf_grid.predict(x_test))
accuracy_score(y_train, cv_rf_clf_grid.predict(x_train))

# plotting best tree
tree.plot_tree(cv_rf_clf_grid.estimators_[29], feature_names = x.columns,class_names=['0', "1"],filled=True)
 
# checking and plotting best features 

imp_df = pd.DataFrame({
    "Varname": x_train.columns,
    "Imp": cv_rf_clf_grid.feature_importances_
})

imp_df.sort_values(by="Imp", ascending=False)
 
plt.plot(imp_df.Varname ,imp_df.Imp , "ro-");plt.xticks(rotation = 90)

