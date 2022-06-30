import numpy as np
import pandas as pd
import statsmodels.tools as sm
from statsmodels.discrete.discrete_model import Logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import scikitplot as skplt
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier, Pool


def pre_process(data: pd.DataFrame):
    data = data.drop(columns=['id']) 
    data['intensity'] = np.sqrt(data['x_mean']**2+data['y_mean']**2+data['z_mean']**2)/10
    data['magnitude'] = (data['x_mean']+data['y_mean']+data['z_mean'])/10
    data['stdovermean1'] = data['x_mean']/data['s1_std']
    data['stdovermean2'] = data['y_mean']/data['s1_std']
    data['stdovermean3'] = data['z_mean']/data['s1_std']
    return data

def corr_matrix(data: pd.DataFrame):
    # Create correlation matrix with the original data
    corr_matrix = data.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.75
    to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
    # Drop features 
    data = data.drop(data[to_drop], axis=1)
    return data

def create_features(data: pd.DataFrame):
    data['Target'] = np.where((data['Target']=='Sitting') | (data['Target']=='Standing') | (data['Target']=='No activity'), 1, 0)
    data = pd.get_dummies(data, drop_first=True)
    Y = data["Target"]
    X = data.drop("Target",axis=1)
    #adding constant
    X = sm.add_constant(X)

    #As the data set suffered from high-dimensionality it was divided into 
    #training and testing sets for validating the model accuracy on test data as well. 
    #The splitting was done manually to avoid random splits that would include the same person both in train and test.
    #splitting the data manually
    y_test = Y[:217]
    y_train = Y[-2191:]
    x_test = X[:217]
    x_train = X[-2191:]
    return X, Y, x_train, x_test, y_train, y_test


def predict_logit(x_train, x_test, y_train, y_test):
    model = Logit(y_train,x_train)
    results = model.fit()
    y_test_hat = results.predict(x_test)
    y_train_hat = results.predict(x_train)
    y_test_class = np.where(y_test_hat>=0.5,1,0)
    y_train_class = np.where(y_train_hat>=0.5,1,0)
    #print("ROC_AUC Train:",roc_auc_score(y_train,y_train_hat).round(2))
    #print("ROC_AUC Test:",roc_auc_score(y_test,y_test_hat).round(2))
    #print("")
    #confusion matrix for the train data
    cm_train=confusion_matrix(y_train,y_train_class).T
    print("Accuracy_train:", round((cm_train[0,0]+cm_train[1,1])/len(y_train),2))
    print("Sensitivity_train:", round(cm_train[1,1]/(cm_train[1,1]+cm_train[0,1]),2))
    print("Specificity_train:", round(cm_train[0,0]/(cm_train[0,0]+cm_train[1,0]),2))
    print("")
    #confusion matrix for the test data
    cm_test=confusion_matrix(y_test,y_test_class).T
    #let's calculate overall accuracy, recall and specificity for test data
    print("Accuracy_test:", round((cm_test[0,0]+cm_test[1,1])/len(y_test),2))
    print("Sensitivity_test:", round(cm_test[1,1]/(cm_test[1,1]+cm_test[0,1]),2))
    print("Specificity_test:", round(cm_test[0,0]/(cm_test[0,0]+cm_test[1,0]),2))
    print("")
    #calculating using classification report
    print("Train:")
    print(classification_report(y_train,y_train_class))
    print("Test:")
    print(classification_report(y_test,y_test_class))
    print("")
    results_summary = pd.DataFrame({"Accuracy":[accuracy_score(y_train,y_train_class),accuracy_score(y_test,y_test_class)],
                              "ROC_AUC":[roc_auc_score(y_train,y_train_hat), roc_auc_score(y_test, y_test_hat)],
                              "Recall":[recall_score(y_train, y_train_class), recall_score(y_test, y_test_class)]
                              },
                             index=["Training set","Testing set"])
    print(results_summary)




def pedict_logit_best(X,Y, x_train, x_test, y_train, y_test):
    #building GridSearch with Logistic Regression
    logit = LogisticRegression(random_state=1)

    param_logit = {"class_weight":["balanced",None],
                   "C":np.linspace(0.01,50,1),
                   'max_iter':[1000]}
                 
    gs_logit = GridSearchCV(estimator = logit,
                          param_grid = param_logit,
                          scoring = "roc_auc", cv=5, verbose=1, n_jobs=2).fit(x_train,y_train) 

    dict_best = gs_logit.best_params_
    #building logit with best params and seeing the results
    logit_grid = LogisticRegression(C = dict_best['C'],
                                    class_weight = dict_best['class_weight'],
                                    max_iter = 1000,
                                    random_state = 1).fit(x_train,y_train)
    
    y_train_hat = logit_grid.predict_proba(x_train)[:,1]
    y_test_hat = logit_grid.predict_proba(x_test)[:,1]
    print("ROC_AUC Train for tuned Logit:",roc_auc_score(y_train,y_train_hat).round(2))
    print("ROC_AUC Test for tuned Logit:",roc_auc_score(y_test,y_test_hat).round(2))
    print("Mean 5-fold ROC AUC score for Tuned Logit",np.mean(cross_val_score(estimator=logit_grid, X=X, y=Y, cv=5, scoring="roc_auc")).round(2))
          


def predict_dt(X,Y,x_train, x_test, y_train, y_test):

    #setting up parameters for DT's GridSearch
    param_dt={"max_depth":range(1,15),
          "min_samples_leaf":range(10,150,10),
          "class_weight":["balanced",None]        
            }
    #fitting GridSearch with above specified parameters
    gs_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=1), param_grid=param_dt,
                  scoring="roc_auc",cv=5)
    gs_dt.fit(x_train,y_train)
    dict_best = gs_dt.best_params_

    dt_grid=DecisionTreeClassifier(class_weight = dict_best['class_weight'],
                                   max_depth = dict_best['max_depth'],
                                   min_samples_leaf = dict_best['min_samples_leaf'],
                                   random_state=1).fit(x_train,y_train)
    
    y_train_hat = dt_grid.predict_proba(x_train)[:,1]
    y_test_hat = dt_grid.predict_proba(x_test)[:,1]

    print("ROC_AUC Train for tuned DT:",roc_auc_score(y_train,y_train_hat).round(2))
    print("ROC_AUC Test for tuned DT:",roc_auc_score(y_test,y_test_hat).round(2))
    print("Mean 5-fold ROC AUC score for Tuned DT",np.mean(cross_val_score(estimator=dt_grid, X=X, y=Y, cv=5, scoring="roc_auc")).round(2))


def predict_RF(X,Y,x_train, x_test, y_train, y_test):

    #setting up parameters 
    param_rf={"max_depth":range(1,15),
              "min_samples_leaf":range(10,150,10),
              "class_weight":["balanced",None]  }
          
    #fitting GridSearch with above specified parameters
    gs_rf = GridSearchCV(estimator=RandomForestClassifier(random_state=1), param_grid=param_rf,
                  scoring="roc_auc", cv=3, n_jobs=2)
    gs_rf.fit(x_train,y_train)

    dict_best = gs_rf.best_params_

    rf_grid = RandomForestClassifier(class_weight = dict_best['class_weight'],
                                     max_depth = dict_best['max_depth'],
                                     min_samples_leaf = dict_best['min_samples_leaf'],
                                     random_state=1).fit(x_train,y_train)
                                   
    y_train_hat = rf_grid.predict_proba(x_train)[:,1]
    y_test_hat = rf_grid.predict_proba(x_test)[:,1]

    print("ROC_AUC Train for tuned RF:",roc_auc_score(y_train,y_train_hat).round(2))
    print("ROC_AUC Test for tuned RF:",roc_auc_score(y_test,y_test_hat).round(2))
    print("Mean 5-fold ROC AUC score for Tuned RF",np.mean(cross_val_score(estimator=rf_grid, X=X, y=Y, cv=5, scoring="roc_auc")).round(2))

    y_test_hat = rf_grid.predict(x_test)
    y_train_hat = rf_grid.predict(x_train)
    y_test_class = np.where(y_test_hat>=0.5,1,0)
    y_train_class = np.where(y_train_hat>=0.5,1,0)
    #calculating using classification report
    print("Random Forest Train:")
    print(classification_report(y_train,y_train_class))

    print("Random Forest Test:")
    print(classification_report(y_test,y_test_class))

    return rf_grid


def plot_aoc_auc(x_test, y_test, model):

    FPR, TPR, thresholds = roc_curve(y_test, model.predict_proba(x_test)[:,1])

    #plotting ROC AUC for Random Forest
    plt.plot(FPR, TPR, label=f"ROC AUC{roc_auc_score(y_test,model.predict_proba(x_test)[:,1]).round(2)}")
    plt.plot([0,1],[0,1])
    plt.legend(loc="lower right")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.show()

    #gain and lift curve plots for the Random Forest
    skplt.metrics.plot_cumulative_gain(y_test, model.predict_proba(x_test))
    plt.show()

    skplt.metrics.plot_lift_curve(y_test, model.predict_proba(x_test))
    plt.show()

def create_catboost_features(data: pd.DataFrame):
    data.loc[data.Target == 'Mixed walking  with little pauses ', 'Target'] = 0
    data.loc[data.Target == 'Walking', 'Target'] = 9
    data.loc[data.Target == 'Mixed walking', 'Target'] = 10
    data.loc[data.Target == 'Mixed walking  includes stairs up ', 'Target'] = 11
    data.loc[data.Target == 'Climbing down', 'Target'] = 5
    data.loc[data.Target == 'Climbing up', 'Target'] = 6
    data.loc[data.Target == 'Stairs up', 'Target'] = 7
    data.loc[data.Target == 'Stairs down', 'Target'] = 8
    data.loc[data.Target == 'Jogging', 'Target'] = 1
    data.loc[data.Target == 'Sitting', 'Target'] = 2
    data.loc[data.Target == 'Standing', 'Target'] = 3
    data.loc[data.Target == 'No activity', 'Target'] = 4

    data = data.astype({'y_max': float, 'z_max': float, 's1_min': float, 'x_min': float, 'z_min': float,
                                 'x_peak_len_20_150': float})
    data = data.astype({'Target': int})
    Y = data["Target"]
    X = data.drop("Target",axis=1)
    y_test = Y[:217]
    y_train = Y[-2191:]
    x_test = X[:217]
    x_train = X[-2191:]
    return X, Y, x_train, x_test, y_train, y_test


def catboost_model(x_train, y_train, x_test, y_test):
    #indicating which features are not categorical
    categorical_features_indices = np.where(x_train.dtypes != np.float)[0]
    train_pool = Pool(x_train, y_train, cat_features=categorical_features_indices)
    #model
    cat = CatBoostClassifier()
    cat.fit(train_pool)
    #make prediction for evaluation
    y_pred_cat = cat.predict(x_test)

    #accuracy
    print('Accuracy ' + str(accuracy_score(y_test, y_pred_cat)))

    return cat

def create_catboost_few_features(data: pd.DataFrame):
    data.loc[data.Target == 'Mixed walking  with little pauses ', 'Target' ] = 'walking'
    data.loc[data.Target == 'Walking', 'Target' ] = 'walking'
    data.loc[data.Target == 'Mixed walking', 'Target' ] = 'walking'
    data.loc[data.Target == 'Mixed walking  includes stairs up ', 'Target' ] = 'walking'

    data.loc[data.Target == 'Climbing down', 'Target' ] = 'down'
    data.loc[data.Target == 'Stairs down', 'Target' ] = 'down'

    data.loc[data.Target == 'Climbing up', 'Target' ] = 'up'
    data.loc[data.Target == 'Stairs up', 'Target' ] = 'up'

    data.loc[data.Target == 'Jogging', 'Target' ] = 'jogging'

    data.loc[data.Target == 'Sitting', 'Target' ] = 'passive'
    data.loc[data.Target == 'Standing', 'Target' ] = 'passive'
    data.loc[data.Target == 'No activity', 'Target' ] = 'passive'

    data = data.astype({'y_max': float, 'z_max': float, 's1_min': float, 'x_min': float, 'z_min': float,
                                      'x_peak_len_20_150': float})
    Y = data["Target"]
    X = data.drop("Target",axis=1)
    y_test = Y[:217]
    y_train = Y[-2191:]
    x_test = X[:217]
    x_train = X[-2191:]
    return X, Y, x_train, x_test, y_train, y_test