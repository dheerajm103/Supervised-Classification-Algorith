import pandas as pd                                            # importing library                  
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("Zoo.csv")                                 # importing dataset

# data cleansing and eda part*****************************************************************************************

df = df.drop(["animal name"], axis = 1)
df.info()                                                     # checking for data types and null values
df.describe()                                                 # checking mean , median and sd
df.duplicated().sum()                                         # checking for duplicate records and removing
df = df.drop_duplicates()
df.var()                                                      # checking for variance

def norm1(i):                                                 # normalization for scaling
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)
norm=norm1(df.iloc[:,0:16])

sns.pairplot(norm)                                            # pairplot for correlation
plt.boxplot(norm)                                             # boxplot for outliers

x = norm                                                      # predictors
y = df.iloc[:,[16]]                                            # target

# model building *************************************************************************************************************
# splitting dataset for training and testing part

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.25)

knn = KNeighborsClassifier(n_neighbors = 3)                  # initialising and fitting the model
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)                                   # test accuracy
pred
print(accuracy_score(Y_test, pred))

pred_train = knn.predict(X_train)                            #  train accuracy
print(accuracy_score(Y_train, pred_train))

# tunning part **************************************************************************************************************
# uswer defined function to get k value

acc = []

for i in range(2,25,1):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    pred_test = neigh.predict(X_test)
    train_acc = accuracy_score(Y_test,pred_test)
    pred_train = neigh.predict(X_train)
    test_acc = accuracy_score(Y_train,pred_train)
    acc.append([train_acc, test_acc])


plt.plot(np.arange(2,25,1),[i[0] for i in acc],"ro-")           # train accuracy plot 

plt.plot(np.arange(2,25,1),[i[1] for i in acc],"bo-")           # test accuracy plot
