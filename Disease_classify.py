
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

os.chdir("E:\\PythonSpace\\Python Machine Learning")
DISEASE_PATH = os.getcwd() + "\\datasets\\heart-disease-uci"
diease_data = pd.read_csv(DISEASE_PATH+'\\heart.csv')

'////////////////////////setting path and load data///////////////////////////'


'------------------------------Visualization----------------------------------'

diease_data.describe()  #Generate various summary statistics
diease_data.corr()  #Compute pairwise correlation of columns

#plot the variables and use multiple axis labels to visualize their values
from pandas.plotting import parallel_coordinates
from matplotlib import pyplot as plt
plt.figure(figsize=(10,5))
parallel_coordinates(diease_data, 'target',color=('#556270', '#4ECDC4'))
plt.show()

#scatters plot
from pandas.plotting import scatter_matrix
attributes = ['age', 'trestbps', 'chol', 'thalach',  'oldpeak']
scatter_matrix(diease_data[attributes], figsize=(10, 5))

'//////////////////////////////Visualization//////////////////////////////////'


'-----------------------------data processing---------------------------------'

#split dataset to training set and testing set sampling
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(diease_data, test_size=0.2, random_state=42)

diease_data_train_X = train_set.drop("target", axis=1)
diease_data_train_y = train_set["target"].copy()

diease_data_test_X = test_set.drop("target", axis=1)
diease_data_test_y = test_set["target"].copy()

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


'------------------------training test and validation-------------------------'

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

#display_score(sgd_accuracy_scores)
#display_score(svm_accuracy_scores)
#display_score(rf_accuracy_scores)

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

#display_sensitivity_and_specificity(sgd_confusion_matrix)
#display_sensitivity_and_specificity(svm_confusion_matrix)
#display_sensitivity_and_specificity(rf_confusion_matrix)

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
plt.title('ROC curve for training data')
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
sgd_importance = eli5.explain_weights(sgd_perm, feature_names = diease_data_train_X_scaled_df.columns.tolist())

svm_perm = PermutationImportance(svm_predictor, random_state=1).fit(diease_data_train_X_scaled, diease_data_train_y)
svm_importance = eli5.explain_weights(svm_perm, feature_names = diease_data_train_X_scaled_df.columns.tolist())

rf_perm = PermutationImportance(rf_predictor, random_state=1).fit(diease_data_train_X_scaled, diease_data_train_y)
rf_importance = eli5.explain_weights(rf_perm, feature_names = diease_data_train_X_scaled_df.columns.tolist())



#plots the features in descending order of relative importance

def plot_feat_importance(columns, feature_importances_, *topx):
    list_feat = list(zip(columns, feature_importances_))
    pd_list_feat = pd.DataFrame(list_feat)
    pd_list_feat.columns = ('Feature','Importance')
    pd_list_feat = pd_list_feat.sort_values(by='Importance')
    pd_list_feat = pd_list_feat[pd_list_feat['Importance']>0]

    if topx:
        pd_list_top = pd_list_feat.iloc[topx[0]:]
    else:
        pd_list_top = pd_list_feat
    plt.figure(figsize=(10,10))
    plt.scatter(y = range(len(pd_list_top)), x = pd_list_top['Importance'])
    plt.yticks(range(len(pd_list_top)),pd_list_top['Feature'])
    plt.title("Relative feature importance of features(training data)", ha = 'center')
    plt.xlabel("Relative feature importance")
    plt.grid(True)
    plt.rcParams['font.size'] = 12
    plt.legend(loc='upper center')
    plt.show()
    return pd_list_top

sgd_list_top = plot_feat_importance(diease_data_train_X_scaled_df.columns, sgd_perm.feature_importances_)
svm_list_top = plot_feat_importance(diease_data_train_X_scaled_df.columns, svm_perm.feature_importances_)
rf_list_top = plot_feat_importance(diease_data_train_X_scaled_df.columns, rf_perm.feature_importances_)

'////////////////////////Training test and validation/////////////////////////'


'---------------------------Performance Measures------------------------------'

#all in one package

from sklearn import metrics
from sklearn.metrics import confusion_matrix

def performance(y, y_pred, print_ = 1, *args):   
    """ Calculate performance measures for a given ground truth classification y and predicted 
    probabilities y_pred. If *args is provided a predifined threshold is used to calculate the performance.
    If not, the threshold giving the best mean sensitivity and specificity is selected. The AUC is calculated
    for a range of thresholds using the metrics package from sklearn. """

    # xx and yy values for ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(y, y_pred, pos_label=1)
    # area under the ROC curve
    AUC = metrics.auc(fpr, tpr)

    if args:
        threshold = args[0]
    else:
        # we will choose the threshold that gives the best balance between sensitivity and specificity
        difference = abs((1-fpr) - tpr)
        threshold = thresholds[difference.argmin()]        
        
    # transform the predicted probability into a binary classification
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # print the performance and plot the ROC curve    
    if print_ == 1:
        print('Threshold: ' + str(round(threshold,2)))
        print('TP: ' + str(tp))
        print('TN: ' + str(tn))
        print('FP: ' + str(fp))
        print('FN: ' + str(fn))
        print("Accuracy: " + str( round(accuracy, 2 )))
        print('Sensitivity: ' + str(round(sensitivity,2)))
        print('Specificity: ' + str(round(specificity,2)))                
        print('AUC: ' + str(round(AUC,2)))
        
        from sklearn.metrics import roc_curve #for model evaluation
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        plt.plot(fpr, tpr, label = 'Classifier', zorder = 1)
        ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.rcParams['font.size'] = 12
        plt.title('ROC curve')
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.grid(True)
        plt.legend(loc='upper right')
        plt.show()
        
    return threshold, AUC, sensitivity, specificity

#integrate perfomance and ROC together
def model_evaluation(model, X_train, y_train, X_test, y_test, print_):
    
    # tune - parameter estimation 
    print('TRAINING SET')
    y_pred_prob_train = model.predict_proba(X_train)
    threshold, AUC_train, sens_train, spec_train = performance(y_train, np.delete(y_pred_prob_train, 0, 1), print_)

    # test
    print('TEST SET')
    y_pred_prob_test = model.predict_proba(X_test)
    _, AUC_test, sens_test, spec_test = performance(y_test, np.delete(y_pred_prob_test, 0, 1), print_, threshold)
    
    # save the results
    results_train = pd.DataFrame(data = [[threshold, AUC_train, sens_train, spec_train, X_train.shape[1]]],
                           columns = ['Threshold','AUC', 'Sensitivity', 'Specificity', '# features'])

    results_test = pd.DataFrame(data = [[threshold, AUC_test, sens_test, spec_test, X_train.shape[1]]],
                           columns = ['Threshold','AUC', 'Sensitivity', 'Specificity', '# features'])
        
    return results_train, results_test, y_pred_prob_train, y_pred_prob_test

sgd_results_train, sgd_results_test, sgd_y_pred_prob_train, sgd_y_pred_prob_test = model_evaluation(sgd_predictor, diease_data_train_X_scaled, diease_data_train_y, diease_data_test_X, diease_data_test_y, print_=1)
svm_results_train, svm_results_test, svm_y_pred_prob_train, svm_y_pred_prob_test = model_evaluation(svm_predictor, diease_data_train_X_scaled, diease_data_train_y, diease_data_test_X, diease_data_test_y, print_=1)
rf_results_train, rf_results_test, rf_y_pred_prob_train, rf_y_pred_prob_test = model_evaluation(rf_predictor, diease_data_train_X_scaled, diease_data_train_y, diease_data_test_X, diease_data_test_y, print_=1)

#summarizes the performance of several classifiers and their ability to generalize
from IPython import display
all_results_train = pd.DataFrame()
all_results_test = pd.DataFrame()

all_results_train = all_results_train.append(sgd_results_train.rename(index={sgd_results_train.index[-1]: 'sgd'}))
all_results_test = all_results_test.append(sgd_results_test.rename(index={sgd_results_test.index[-1]: 'sgd'}))
all_results_train.loc['sgd', '#features'] = len(sgd_list_top)
all_results_test.loc['sgd', '#features'] = len(sgd_list_top)
                      
all_results_train = all_results_train.append(svm_results_train.rename(index={svm_results_train.index[-1]: 'svm'}))
all_results_test = all_results_test.append(svm_results_test.rename(index={svm_results_test.index[-1]: 'svm'}))
all_results_train.loc['svm', '#features'] = len(svm_list_top)
all_results_test.loc['svm', '#features'] = len(svm_list_top)
                     
all_results_train = all_results_train.append(rf_results_train.rename(index={rf_results_train.index[-1]: 'rf'}))
all_results_test = all_results_test.append(rf_results_test.rename(index={rf_results_test.index[-1]: 'rf'}))
all_results_train.loc['rf', '#features'] = len(rf_list_top)
all_results_test.loc['rf', '#features'] = len(rf_list_top)
           
print('Performance in training set')
display.display(np.round(all_results_train, decimals = 2))
print()

print('Performance in test set')
display.display(np.round(all_results_test, decimals = 2))
print()

print('Performance in test set')
display.display(np.round(all_results_test, decimals = 2))
print()

diff = all_results_train-all_results_test
diff[['AUC', 'Sensitivity','Specificity']].plot(kind = 'bar', figsize = (5,5))
plt.ylabel('Difference in performance')
plt.xticks(rotation=None)
plt.title('Difference in performance (training - test)')
plt.show()

'/////////////////////////Performance Measures/////////////////////////////////'























