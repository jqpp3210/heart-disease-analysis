
'---------------------------------Config--------------------------------------'

Min_Max_Scaling_Or_Not = False
Standard_Scaling_Or_Not = True

import warnings
warnings.filterwarnings('ignore')

'/////////////////////////////////Config//////////////////////////////////////'


'------------------------setting path and load data---------------------------'

import numpy as np # linear algebra package
import pandas as pd # for data processing, CSV file I/O (e.g. pd.read_csv)
import os

#setting path and load data

os.chdir("D:\\Python Space\\Python Machine Learning")
DISEASE_PATH = os.getcwd() + "\\datasets\\heart-disease-uci"
diease_data = pd.read_csv(DISEASE_PATH+'\\heart.csv')

'////////////////////////setting path and load data///////////////////////////'


'------------------------------Visualization----------------------------------'

diease_data.describe()  #Generate various summary statistics
diease_data.corr()  #Compute pairwise correlation of columns

#plot the variables and use multiple axis labels to visualize their values
from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
plt.figure(figsize=(20,10))
parallel_coordinates(diease_data, 'target',color=('#556270', '#4ECDC4'))
plt.show()

#scatters plot
from pandas.plotting import scatter_matrix
attributes = ['age', 'trestbps', 'chol', 'thalach',  'oldpeak']
scatter_matrix(diease_data[attributes], figsize=(12, 8))

'//////////////////////////////Visualization//////////////////////////////////'


'-----------------------------data processing---------------------------------'

#split dataset to training set and testing set sampling
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(diease_data, test_size=0.2, random_state=42)

diease_data_train_X = train_set.drop("target", axis=1)
diease_data_train_y = train_set["target"].copy()

#scaling
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin

class Scaling_Check(BaseEstimator, TransformerMixin):
    def __init__(self, Min_Max_Scaling_Or_Not = True, Standard_Scaling_Or_Not = True): # no *args or **kargs
        self.Min_Max_Scaling_Or_Not = Min_Max_Scaling_Or_Not
        self.Standard_Scaling_Or_Not = Standard_Scaling_Or_Not
    
    def fit(self, X, y = None):
        return self  # nothing else to do
    
    def transform(self, X, y = None):
        diease_data_scaled = X
        if self.Min_Max_Scaling_Or_Not:
            min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            diease_data_scaled = min_max_scaler.fit_transform(diease_data_scaled)
        if self.Standard_Scaling_Or_Not:
            standard_scaler = preprocessing.StandardScaler(copy=True, with_mean=True, with_std=True)
            diease_data_scaled = standard_scaler.fit_transform(diease_data_scaled)
            
        return diease_data_scaled

scaling_checker = Scaling_Check(Min_Max_Scaling_Or_Not,Standard_Scaling_Or_Not)
diease_data_train_X_scaled = scaling_checker.transform(diease_data_train_X)
'/////////////////////////////data processing/////////////////////////////////'


'---------------------------------modeling-------------------------------------'

from sklearn.linear_model import SGDClassifier
sgd_predictor = SGDClassifier(loss='log',random_state=42) #loss='log' make it a probabilistic classifier
sgd_predictor.fit(diease_data_train_X_scaled, diease_data_train_y)
sgd_predictions = sgd_predictor.predict(diease_data_train_X_scaled)

from sklearn import svm
svm_predictor = svm.SVC(kernel='rbf', probability=True) #probability=true make it a probabilistic classifier
svm_predictor.fit(diease_data_train_X_scaled, diease_data_train_y)
svm_predictions = svm_predictor.predict(diease_data_train_X_scaled)

from sklearn.ensemble import RandomForestClassifier
rf_predictor = RandomForestClassifier(max_depth=5)
rf_predictor.fit(diease_data_train_X_scaled, diease_data_train_y)
rf_predictions = rf_predictor.predict(diease_data_train_X_scaled)

'/////////////////////////////////modeling/////////////////////////////////////'


'--------------------------Performance Measures-------------------------------'

"""
from sklearn.metrics import mean_squared_error  #mean_squared_error
sgd_mse = mean_squared_error(diease_data_train_y, sgd_predictions)
sgd_rmse = np.sqrt(sgd_mse)
"""

#Measuring Accuracy Using Cross-Validation
from sklearn.model_selection import cross_val_score  #cross-validation
sgd_scores = cross_val_score(sgd_predictor, diease_data_train_X_scaled, 
                              diease_data_train_y, scoring="accuracy",
                              cv=5)

svm_scores = cross_val_score(svm_predictor, diease_data_train_X_scaled, 
                              diease_data_train_y, scoring="accuracy",
                              cv=5)

rf_scores = cross_val_score(rf_predictor, diease_data_train_X_scaled, 
                              diease_data_train_y, scoring="accuracy",
                              cv=5)

sgd_accuracy_scores = np.sqrt(sgd_scores)
svm_accuracy_scores = np.sqrt(svm_scores)
rf_accuracy_scores = np.sqrt(rf_scores)

def display_score(list):
    print(','.join([str(i.round(2)) for i in list]))

display_score(sgd_accuracy_scores)
display_score(svm_accuracy_scores)
display_score(rf_accuracy_scores)

#Confusion Matrix
from sklearn.model_selection import cross_val_predict
sgd_train_pred = cross_val_predict(sgd_predictor, diease_data_train_X_scaled, 
                                   diease_data_train_y, cv=5)

svm_train_pred = cross_val_predict(svm_predictor, diease_data_train_X_scaled, 
                                   diease_data_train_y, cv=5)

rf_train_pred = cross_val_predict(rf_predictor, diease_data_train_X_scaled, 
                                   diease_data_train_y, cv=5)
'instead of returning the evaluation scores, it returns the predictions made on each test fold'

def display_sensitivity_and_specificity(confusion_matrix):
    sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[1,0])
    print('Sensitivity : ', sensitivity )
    specificity = confusion_matrix[1,1]/(confusion_matrix[1,1]+confusion_matrix[0,1])
    print('Specificity : ', specificity)

from sklearn.metrics import confusion_matrix
sgd_confusion_matrix = confusion_matrix(diease_data_train_y, sgd_train_pred)
svm_confusion_matrix = confusion_matrix(diease_data_train_y, svm_train_pred)
rf_confusion_matrix = confusion_matrix(diease_data_train_y, rf_train_pred)

display_sensitivity_and_specificity(sgd_confusion_matrix)
display_sensitivity_and_specificity(svm_confusion_matrix)
display_sensitivity_and_specificity(rf_confusion_matrix)

"""
True-Positives   False-Positives
False-Positives  True-Negatives
"""

#ROC Curve
from sklearn.metrics import roc_curve #for model evaluation
sgd_y_pred_quant = sgd_predictor.predict_proba(diease_data_train_X_scaled)[:, 1]
sgd_fpr, sgd_tpr, sgd_thresholds = roc_curve(diease_data_train_y, sgd_y_pred_quant)

svm_y_pred_quant = svm_predictor.predict_proba(diease_data_train_X_scaled)[:, 1]
svm_fpr, svm_tpr, svm_thresholds = roc_curve(diease_data_train_y, svm_y_pred_quant)

rf_y_pred_quant = rf_predictor.predict_proba(diease_data_train_X_scaled)[:, 1]
rf_fpr, rf_tpr, rf_thresholds = roc_curve(diease_data_train_y, rf_y_pred_quant)

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
sgd_plot = plt.plot(sgd_fpr, sgd_tpr, label = "sgd")
svm_plot = plt.plot(svm_fpr, svm_tpr, label = "svm")
rf_plot = plt.plot(rf_fpr, rf_tpr, label = "rf")

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for heart disease classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.legend(loc='upper right')

#Another common metric is the Area Under the Curve(AUC)
from sklearn.metrics import auc
auc(sgd_fpr, sgd_tpr)
auc(svm_fpr, svm_tpr)
auc(rf_fpr, rf_tpr)

#Permutation importance

diease_data_train_X_scaled_df = pd.DataFrame(diease_data_train_X_scaled)
temp = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
       'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
diease_data_train_X_scaled_df.columns = temp

import eli5 #for purmutation importance
from eli5.sklearn import PermutationImportance
sgd_perm = PermutationImportance(sgd_predictor, random_state=1).fit(diease_data_train_X_scaled, diease_data_train_y)
importance = eli5.explain_weights(sgd_perm, feature_names = diease_data_train_X_scaled_df.columns.tolist())
print(eli5.format_as_text(importance))


'//////////////////////////Performance Measures///////////////////////////////'






























