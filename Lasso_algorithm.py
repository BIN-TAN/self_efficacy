import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
t = time.process_time()

##############################################################################method 3: use for-loop
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
#import data
data = pd.read_csv("597final.csv")

#define target
y = data["self_efficacy"]
y = y * 15 + 100
data.drop("self_efficacy", axis=1, inplace = True)

#define features
X = data.iloc[:, 1:]
X.drop(['gender', 'fixed_mindset', 'scared', 'lively', 'miserable', 'proud', 'afraid', 'sad'], axis=1, inplace=True)
##############################################################################

# configure the cross-validation procedure
cv_outer = KFold(n_splits=4, shuffle=True, random_state=99)

# enumerate splits
outer_results = list()
inner_results = list()
feature_importance_results = pd.DataFrame(list())
for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :] 
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix] 

    cv_inner = KFold(n_splits=4, shuffle=True, random_state=99) 
    model = Lasso() 

    space = dict() 
    space['alpha'] = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1] 

    search = GridSearchCV(model, space, 
                          scoring = ["r2", "neg_mean_squared_error",
                                     "neg_mean_absolute_error"], 
                          refit= "r2", cv=cv_inner,
                          return_train_score=True
                          ) 

    results = search.fit(X_train, y_train) 
    best_model = results.best_estimator_ 

#get inner results:
    inner_results.append(results.cv_results_)

    # Extract the best hyperparameters' index
    best_idx = results.best_index_

    # Extract training and validation scores for R2, RMSE, and MAE for the best hyperparameters
    best_train_r2 = results.cv_results_['mean_train_r2'][best_idx]
    best_valid_r2 = results.cv_results_['mean_test_r2'][best_idx]

    best_train_rmse = np.sqrt(-results.cv_results_['mean_train_neg_mean_squared_error'][best_idx])
    best_valid_rmse = np.sqrt(-results.cv_results_['mean_test_neg_mean_squared_error'][best_idx])

    best_train_mae = -results.cv_results_['mean_train_neg_mean_absolute_error'][best_idx]
    best_valid_mae = -results.cv_results_['mean_test_neg_mean_absolute_error'][best_idx]

    inner_results.append({
        'best_train_r2': best_train_r2,
        'best_valid_r2': best_valid_r2,
        'best_train_rmse': best_train_rmse,
        'best_valid_rmse': best_valid_rmse,
        'best_train_mae': best_train_mae,
        'best_valid_mae': best_valid_mae,
    })
    
#outer evaluation and get the results
    yhat = best_model.predict(X_test) 
    rmse = math.sqrt(mean_squared_error(y_test, yhat))
    mae = mean_absolute_error(y_test, yhat) 
    r2 = r2_score(y_test, yhat)
    outer_results.append([rmse, mae, r2]) 
    
    print('>results=%.3f, best_score=%.3f, best_param=%s' % (r2, results.best_score_, results.best_params_))

#get the feature importance
    importance = np.transpose(best_model.coef_)
    feat_importances = pd.DataFrame(np.abs(importance), index = X_train.columns)
    feature_importance_results = pd.concat([feature_importance_results, feat_importances], axis = 1)

results_lasso_outer = pd.DataFrame(outer_results)
results_lasso_outer.to_csv("resultslasso_outer.csv")

results_lasso_inner = pd.DataFrame(inner_results)
results_lasso_inner.to_csv("resultslasso_inner.csv")

lasso_feature_importance = pd.DataFrame(feat_importances)
feat_importances.to_csv("lasso_feature_importance_rank.csv")
lasso_feature_importance.to_csv("lasso_feature_importance.csv")

elapsed_time = time.process_time() - t
print(elapsed_time)
