import pandas as pd                             # importing library
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("loan.csv")                   # importing dataset

# data cleansing and eda part **************************************************
# dropping nominal columns
df = df.iloc[:,[16,2,3,4,7,13,24,32,34,36,37,38,39,40,41,42,43,44,46]]           
df.duplicated().sum()                           # checking for duplicate records
df.info()                                       # cheking for data types and null values
df.describe()                                   # checking for mean , median and sd
df.loan_status.value_counts()                   # checking unique values for dependent variables
df.skew()                                       # checking skewness
df.kurtosis()                                   # checking kurtosis
df.corr()                                       # checking kurtosis

sns.pairplot(df)                                # pairplot for eda 
sns.pairplot(df, hue = "loan_status")           # pairplot wrt to loan status

# Boxplot of independent variable distribution for each category loan_status 
sns.boxplot(x = "loan_status", y = "loan_amnt", data = df)
sns.boxplot(x = "loan_status", y = "funded_amnt", data = df)
sns.boxplot(x = "loan_status", y = "funded_amnt_inv", data = df)
sns.boxplot(x = "loan_status", y = "installment", data = df)
sns.boxplot(x = "loan_status", y = "annual_inc", data = df)
sns.boxplot(x = "loan_status", y = "dti", data = df)
sns.boxplot(x = "loan_status", y = "revol_bal", data = df)
sns.boxplot(x = "loan_status", y = "total_acc", data = df)


# Scatter plot for each categorical loan_status
sns.stripplot(x = "loan_status", y = "loan_amnt", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "funded_amnt", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "funded_amnt_inv", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "installment", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "annual_inc", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "dti", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "revol_bal", jitter = True, data = df)
sns.stripplot(x = "loan_status", y = "total_acc", jitter = True, data = df)

# model building ****************************************************************
# splitting the data set
train, test = train_test_split(df, test_size = 0.2)

# fitting the model for loan staus  as dependent variable
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, 1:], train.iloc[:,0 ])

test_predict = model.predict(test.iloc[:, 1:])          # Test predictions
confusion_matrix(test.iloc[:,0], test_predict)
accuracy_score(test.iloc[:,0], test_predict)            # Test accuracy 

train_predict = model.predict(train.iloc[:, 1:])        # Train predictions 
confusion_matrix(train.iloc[:,0], train_predict)
accuracy_score(train.iloc[:,0], train_predict)          # Train accuracy


