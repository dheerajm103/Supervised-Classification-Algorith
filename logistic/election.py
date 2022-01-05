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

df = pd.read_csv("election_data.csv", sep = ",")    # importing dataset

# data cleansing and eda part ******************************************************************

df.nunique()                                      # checking unique values
# dropping nominal column
df = df.drop(df.index[0], axis = 0)
df.info()                                         # checking for null values and data type
df.describe()                                     # checking mean ,median and sd
df.duplicated().sum()                             # checking duplicate rows
corr = df.corr()                                  # checking for correlation
df.skew()                                         # checking for skewness
df.kurtosis()                                     # checking for kurtosis
plt.boxplot(df)                                   # boxplot for outliers
sns.pairplot(df)                                  # pair plot for eda
def norm1(i):                                     # normalization
	x = (i-i.min())	/(i.max()-i.min())
	return(x)

norm = norm1(df)
         
# Model building *******************************************************************************

logit_model = smf.logit('Result ~  Electionid +Year + AmountSpent + PopularityRank ', data = norm).fit()

logit_model.summary2()                             #summary for AIC
logit_model.summary()

pred = logit_model.predict(norm.iloc[ :, [0,2,3,4]])

fpr, tpr, thresholds = roc_curve(norm.Result, pred)    # calculating threshold value
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

norm["pred"] = np.zeros(10)                      # comparing threshold values
norm.loc[pred > optimal_threshold, "pred"] = 1

# classification report
classification = classification_report(norm["pred"], norm["Result"])
classification


# Splitting the data into train and test data 
train_data, test_data = train_test_split(norm, test_size = 0.2) 

# Final Model building 
model = smf.logit('Result ~  Electionid +Year + AmountSpent + PopularityRank ', data = train_data).fit()

model.summary2()                                   #summary for AIC
model.summary()

# Prediction on Test data set
test_pred = logit_model.predict(test_data)

test_data["test_pred"] = np.zeros(2)
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['Result'])
confusion_matrix

accuracy_test = (1 + 1)/(2) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["Result"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["Result"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_tes
# prediction on train data
train_pred = model.predict(train_data.iloc[ :, [0,2,3,4]])

train_data["train_pred"] = np.zeros(8)
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['Result'])
confusion_matrx

accuracy_train = (3 + 4)/(8)
print(accuracy_train)
