import pandas as pd                              # importing library
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
import pylab as pl

df = pd.read_csv("Affairs.csv", sep = ",")        # importing dataset

# data cleansing and eda part ******************************************************************
df['naffairs'].unique()                           # checking unique value 
naffairs_uni_count = pd.value_counts(df['naffairs'])
naffairs_uni_count

df = df.drop(["id"], axis = 1)                    # dropping nominal column
df.info()                                         # checking for null values and data type
df.describe()                                     # checking mean ,median and sd
df.duplicated().sum()                             # checking duplicate rows
corr = df.corr()                                  # checking for correlation
df.skew()                                         # checking for skewness
df.kurtosis()                                     # checking for kurtosis
plt.boxplot(df)                                   # boxplot for outliers
sns.pairplot(df)                                  # pair plot for eda

df['naffairs'] = np.where(df['naffairs'] <= 0, 0, df['naffairs'])    
df['naffairs'] = np.where(df['naffairs'] > 0, 1, df['naffairs'])

# Model building *******************************************************************************

logit_model = smf.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = df).fit()

logit_model.summary2()                             #summary for AIC
logit_model.summary()

pred = logit_model.predict(df.iloc[ :, 1:])

fpr, tpr, thresholds = roc_curve(df.naffairs, pred)    # calculating threshold value
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

i = np.arange(len(tpr))                            # calculating roc dataframe
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

fig, ax = pl.subplots()                            # Plot tpr vs 1-fpr
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)                            # accuracy and area under the curve
print("Area under the ROC curve : %f" % roc_auc)

df["pred"] = np.zeros(601)                      # comparing threshold values
df.loc[pred > optimal_threshold, "pred"] = 1

# classification report
classification = classification_report(df["pred"], df["naffairs"])
classification


# Splitting the data into train and test data 
train_data, test_data = train_test_split(df, test_size = 0.3) 

# Final Model building 
model = smf.logit('naffairs ~ kids + vryunhap + unhap + avgmarr + hapavg + vryhap + antirel + notrel + slghtrel + smerel + vryrel + yrsmarr1 + yrsmarr2 + yrsmarr3 + yrsmarr4 + yrsmarr5 + yrsmarr6', data = train_data).fit()

model.summary2()                                   #summary for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

test_data["test_pred"] = np.zeros(181)
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['naffairs'])
confusion_matrix

accuracy_test = (86 + 30)/(181) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["naffairs"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["naffairs"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test

# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

train_data["train_pred"] = np.zeros(420)
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['naffairs'])
confusion_matrx

accuracy_train = (232 + 64)/(420)
print(accuracy_train)
