import pandas as pd                                                   # importing library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.metrics import accuracy_score
import scipy.stats as stats
import pylab
import seaborn as sns

s_train = pd.read_csv("SalaryData_Train.csv")                        # importing dataset
s_test = pd.read_csv("SalaryData_Test.csv")

df = pd.concat([s_train, s_test], axis=0)                            # concatinating to one dataset
df

# data cleansing and eda part*************************************************************************************************

df = df.drop(["educationno"],axis = 1)                                # dropping nominal columns

df.duplicated().sum()                                                 # checking for duplicasy and removing
df = df.drop_duplicates()

df.info()                                                             # checking for data types and null values
df.describe()                                                         # checking mean,median and sd

sns.pairplot(s_train.iloc[:, :])                                      # plotting pair plot for correlation

plt.boxplot(df.iloc[:,[0,8,9,10]])                                    # boxplot for outliers

df.age = np.log10(df.age)                                             # log transformation for normal distribution
df.hoursperweek = np.log(df.hoursperweek)
stats.probplot(df.age, dist="norm", plot=pylab)                       # QQ plot for normal distribution
stats.probplot(df.hoursperweek, dist="norm", plot=pylab) 

df1 = pd.get_dummies(df.iloc[:,0:12] , drop_first = True)             # dummy column for catogrical column
df1

def norm1(i):                                                         # normalization for scaling
    x = (i - i.min())/(i.max() - i.min())
    return x
norm = norm1(df1)
norm
norm["Salary"] = df.Salary

# model building ******************************************************************************************************
# splitting data to train and test

norm_train, norm_test = train_test_split(norm, test_size = 0.25,random_state = 50)
x1 = norm_train.iloc[:,0:93]
x2 = norm_test.iloc[:,0:93]

classifier_mb = MB()                                                   # initialising multinomial NB
classifier_mb.fit(x1, norm_train.Salary)                               # fitting model

test_pred_m = classifier_mb.predict(x2)                                # test accuracy
test_accuracy = accuracy_score(test_pred_m, norm_test.Salary) 
test_accuracy 

train_pred_m = classifier_mb.predict(x1)                               # train accuracy
train_accuracy = accuracy_score(train_pred_m, norm_train.Salary) 
train_accuracy 

pd.crosstab(test_pred_m, norm_test.Salary)                             # confusion matrix
pd.crosstab(train_pred_m, norm_train.Salary)

# tunning ********************************************************************************************************************

classifier_mb_lap = MB(alpha = 3)                                       # laplace transformation intialising                                  
classifier_mb_lap.fit(x1, norm_train.Salary)


test_pred_lap = classifier_mb_lap.predict(x2)                           # laplace test accuracy
accuracy_test_lap = accuracy_score(test_pred_lap, norm_test.Salary) 
accuracy_test_lap

train_pred_lap = classifier_mb_lap.predict(x1)                          # laplace train accuracy
accuracy_train_lap = accuracy_score(train_pred_lap, norm_train.Salary) 
accuracy_train_lap

