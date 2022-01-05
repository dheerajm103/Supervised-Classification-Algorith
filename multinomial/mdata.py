import pandas as pd                             # importing library
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv("mdata.csv")                   # importing dataset

# data cleansing and eda part **************************************************
df = df.drop(["sr","id"], axis = 1)             # dropping nominal columns
df.duplicated().sum()                           # checking for duplicate records
df.info()                                       # cheking for data types and null values
df.describe()                                   # checking for mean , median and sd
df.prog.value_counts()                          # checking unique values for dependent variables
df.ses.value_counts()
df.honors.value_counts()
df.schtyp.value_counts()
df.skew()                                       # checking skewness
df.kurtosis()                                   # checking kurtosis
df.corr()                                       # checking kurtosis
sns.pairplot(df)                                # pairplot for eda 
sns.pairplot(df, hue = "prog")                  # pairplot wrt prog
sns.pairplot(df, hue = "ses")                   # pairplot wrt ses

# Boxplot of independent variable distribution for each category prog and ses 
sns.boxplot(x = "prog", y = "read", data = df)
sns.boxplot(x = "prog", y = "write", data = df)
sns.boxplot(x = "prog", y = "math", data = df)
sns.boxplot(x = "prog", y = "science", data = df)
sns.boxplot(x = "ses", y = "read", data = df)
sns.boxplot(x = "ses", y = "write", data = df)
sns.boxplot(x = "ses", y = "math", data = df)
sns.boxplot(x = "ses", y = "science", data = df)


# Scatter plot for each categorical prog and ses
sns.stripplot(x = "prog", y = "read", jitter = True, data = df)
sns.stripplot(x = "prog", y = "write", jitter = True, data = df)
sns.stripplot(x = "prog", y = "math", jitter = True, data = df)
sns.stripplot(x = "prog", y = "science", jitter = True, data = df)
sns.stripplot(x = "ses", y = "read", jitter = True, data = df)
sns.stripplot(x = "ses", y = "write", jitter = True, data = df)
sns.stripplot(x = "ses", y = "math", jitter = True, data = df)
sns.stripplot(x = "ses", y = "science", jitter = True, data = df)

# model building ****************************************************************
# splitting the data set
train, test = train_test_split(df, test_size = 0.2)

# fitting the model for prog as dependent variable
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, [4,5,6,7]], train.iloc[:,3 ])

test_predict = model.predict(test.iloc[:, [4,5,6,7]])   # Test predictions
confusion_matrix(test.iloc[:,3], test_predict)
accuracy_score(test.iloc[:,3], test_predict)            # Test accuracy 

train_predict = model.predict(train.iloc[:, [4,5,6,7]]) # Train predictions 
confusion_matrix(train.iloc[:,3], train_predict)
accuracy_score(train.iloc[:,3], train_predict)          # Train accuracy


# fitting the model for ses as dependent variable
model = LogisticRegression(multi_class = "multinomial", solver = "newton-cg").fit(train.iloc[:, [4,5,6,7]], train.iloc[:,1 ])

test_predict = model.predict(test.iloc[:, [4,5,6,7]])   # Test predictions
confusion_matrix(test.iloc[:,1], test_predict)
accuracy_score(test.iloc[:,1], test_predict)            # Test accuracy 

train_predict = model.predict(train.iloc[:, [4,5,6,7]]) # Train predictions 
confusion_matrix(train.iloc[:,1], train_predict)
accuracy_score(train.iloc[:,1], train_predict)          # Train accuracy
