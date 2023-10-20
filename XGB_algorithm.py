import warnings
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

import time
t = time.process_time()


##############################################################################method 3: use for-loop
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
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


# Create a histogram
plt.figure(figsize=(10, 6))  # Optional, to adjust the figure size
plt.hist(y, bins=30, color='cadetblue', edgecolor='k', alpha=0.7)

# Labeling
plt.title('Histogram of Preprocessed Self-Efficacy Scores')
plt.xlabel('Preprocessed Self-Efficacy Scores')
plt.ylabel('Frequency')

# Display the histogram
plt.grid(axis='y', alpha=0.75)  # Optional, to add a grid on y-axis
plt.show()


# configure the cross-validation procedure
cv_outer = KFold(n_splits=4, shuffle=True, random_state=99)

# enumerate splits
outer_results = list()
inner_results = list()

for train_ix, test_ix in cv_outer.split(X):
    X_train, X_test = X.iloc[train_ix, :], X.iloc[test_ix, :] 
    y_train, y_test = y.iloc[train_ix], y.iloc[test_ix] 

    cv_inner = KFold(n_splits=4, shuffle=True, random_state=99) 
    model = XGBRegressor(random_state=99) 

    space = dict() 
    space['n_estimators'] = [50, 100, 150] 
    space['max_depth'] = [5, 10, 15, 20, 25] 

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
    importance = best_model.feature_importances_
    feat_importances = pd.Series(importance, index = X_train.columns)
    plots = feat_importances.nlargest(15).plot(kind='barh')
    plt.savefig("plot_xgb.png")

results_xgb_outer = pd.DataFrame(outer_results)
results_xgb_outer.to_csv("resultsxgb_outer.csv")

results_xgb_inner = pd.DataFrame(inner_results)
results_xgb_inner.to_csv("resultsxgb_inner.csv")

xgb_feature_importance = pd.DataFrame(feat_importances)
feat_importances.to_csv("xgb_feature_importance_rank.csv")
xgb_feature_importance.to_csv("xgb_feature_importance_top15.csv")

elapsed_time = time.process_time() - t
print(elapsed_time)

import shap
#explain every prediction
explainer = shap.Explainer(search.best_estimator_)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])
shap.plots.beeswarm(shap_values)
shap.plots.bar(shap_values)
shap.summary_plot(shap_values, plot_type='violin')
