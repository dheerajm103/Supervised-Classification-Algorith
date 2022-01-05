import pandas as pd                                             # importing libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB as GB
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import pylab
import seaborn as sns

df = pd.read_csv("NB_Car_Ad.csv")                                # importing dataset
df

# data cleansing and eda part **********************************************************************************

df = df.drop(["User ID"],axis = 1)                               # dropping nominal column

df.duplicated().sum()                                            # checking and removing duplicate rows
df = df.drop_duplicates()

df.info()                                                        # checking for data types and null values                                       
df.describe()                                                    # checking mean,median and sd

plt.boxplot(df.iloc[:,[1,2]])                                    # boxplot for outliers
sns.pairplot(df.iloc[:, :])                                      # pairplot for correlation
stats.probplot(df.EstimatedSalary, dist="norm", plot=pylab)      # QQ plot for normal distribution
stats.probplot(df.Age, dist="norm", plot=pylab) 

df1 = pd.get_dummies(df , drop_first = True)                     # dummy column for categorical columns
df1
df1.Age = np.log10(df1.Age)                                      # log transformation for normal distribution
df1.EstimatedSalary = np.log10(df1.EstimatedSalary)

def norm1(i):                                                    # normalization for scaling
    x = (i - i.min())/(i.max() - i.min())
    return x
norm = norm1(df1)
norm
df2 = norm.iloc[:, [2,0,1,3] ]  
df2

# model building******************************************************************************************************
# splitting data to train and test

df2_train, df2_test = train_test_split(df2, test_size = 0.25, random_state = 20)
x1 = df2_train.iloc[:,[1,2,3]]
x2 = df2_test.iloc[:,[1,2,3]]

classifier_gb = GB()                                             # initialising guassian NB
classifier_gb.fit(x1,df2_train.Purchased)                        # fitting model

test_pred_g = classifier_gb.predict(x2)                          # test accuracy
test_accuracy = accuracy_score(test_pred_g, df2_test.Purchased) 
test_accuracy 

train_pred_g = classifier_gb.predict(x1)                         # train accuracy
train_accuracy = accuracy_score(train_pred_g, df2_train.Purchased) 
train_accuracy 

pd.crosstab(test_pred_g, df2_test.Purchased)                     # confusion matrix
pd.crosstab(train_pred_g, df2_train.Purchased)



