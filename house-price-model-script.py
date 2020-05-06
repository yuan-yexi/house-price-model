# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# %%
df_raw = pd.read_csv('hdb_resale_model_output_03_Apr.csv')


# %%
df = df_raw.dropna()


# %%
df.isna().sum()

# %% [markdown]
# ## Feature Engineering

# %%
# Separate our numerical and categorical variables
cat_features = ['town', 'flat_type', 'storey_range']
num_features = ['min_dist_mrt', 'min_dist_mall', 'cbd_dist','floor_area_sqm', 'lease_remain_years', ]
target = ['price_per_sqm']


# %%
df_cat = df[cat_features]
df_num = df[num_features]
df_target = df[target]


# %%
# Mapping ordinal categories to their respective values
flat_type_map = {
    'EXECUTIVE': 7,
    'MULTI-GENERATION': 6,
    '5 ROOM': 5,
    '4 ROOM': 4,
    '3 ROOM': 3,
    '2 ROOM': 2,
    '1 ROOM': 1
}

df_cat['flat_type_mapped'] = df_cat['flat_type'].map(lambda x: flat_type_map[x])


# %%
def split_mean(x):
    split_list = x.split(' TO ')
    mean = (float(split_list[0])+float(split_list[1]))/2
    return mean

df_cat['storey_mean'] = df_cat['storey_range'].apply(lambda x: split_mean(x))


# %%
# One-Hot Encoding for'town' and drop 1 of our dummy variables
df_cat = pd.get_dummies(data=df_cat, columns=['town'], drop_first=True)


# %%
# Inspect categorical features
df_cat = df_cat.drop(['flat_type', 'storey_range'], axis=1)
df_cat.info()


# %%
df_num.info()

# %% [markdown]
# ## Train, Test, Split

# %%
# Train, test, splot
X = df_cat.join(df_num)
y = df_target
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# ## Import Evaluation Metrics for ML Modls

# %%
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.model_selection import cross_val_score

# %% [markdown]
# ## Linear Regression

# %%
from sklearn.linear_model import LinearRegression


# %%
lm_reg = LinearRegression().fit(X, y)

# %% [markdown]
# ### Linear Regression Model Evaluation MSE

# %%
# Compute y_pred_lm
y_pred_lm = lm_reg.predict(X_test)

# Compute rmse_lm
mse_lm = MSE(y_test, y_pred_lm)

# Compute rmse_lm
rmse_lm = mse_lm**(1/2)

# Print rmse_lm
print("Test set RMSE of Linear Regression: {:.2f}".format(rmse_lm))


# %%
# Create a pd.Series of features importances
lm_coef = lm_reg.coef_
lm_coef

# %% [markdown]
# ### 10-Fold Validation: Linear Regression

# %%
lm_cv_scores = cross_val_score(lm_reg, X, y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold cross-validation scores
print(lm_cv_scores)
lm_cv_score_rmse = abs(lm_cv_scores)**(1/2)
print(lm_cv_score_rmse)

print("Average 10-Fold CV Score: {}".format(np.mean(lm_cv_score_rmse)))

# %% [markdown]
# ## Regression Tree

# %%
# Import DecisionTreeRegressor from sklearn.tree
from sklearn.tree import DecisionTreeRegressor


# %%
rt = DecisionTreeRegressor(max_depth=8,
             min_samples_leaf=0.13,
            random_state=3)

# Fit dt to the training set
rt.fit(X_train, y_train)

# %% [markdown]
# ### Regression Tree Model Evaluation MSE

# %%
# Compute y_pred
y_pred_rt = rt.predict(X_test)

# Compute mse_dt
mse_rt = MSE(y_test, y_pred_rt)

# Compute rmse_dt
rmse_rt = mse_rt**(1/2)

# Print rmse_dt
print("Test set RMSE of dt: {:.2f}".format(rmse_rt))


# %%
# Create a pd.Series of features importances
rt_importances = pd.Series(data=rt.feature_importances_,
                        index= X_train.columns)

# Sort importances
rt_importances_sorted = rt_importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12,15))
rt_importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# %%
rt_cv_scores = cross_val_score(rt, X, y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold cross-validation scores
print(rt_cv_scores)
rt_cv_score_rmse = abs(rt_cv_scores)**(1/2)
print(rt_cv_score_rmse)

print("Average 10-Fold CV Score: {}".format(np.mean(rt_cv_score_rmse)))

# %% [markdown]
# ## Random Forest Regression

# %%
# Import RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor

# Instantiate rf
rf = RandomForestRegressor(n_estimators=25,
            random_state=2)
            
# Fit rf to the training set    
rf.fit(X_train, y_train)

# %% [markdown]
# ### Random Forest Regressior Model Evaluation MSE

# %%
# Predict the test set labels
y_pred_rf = rf.predict(X_test)

# Evaluate the test set RMSE
rmse_test_rf = (MSE(y_test, y_pred_rf))**(1/2)

# Print rmse_test
print('Test set RMSE of rf: {:.2f}'.format(rmse_test_rf))

# %% [markdown]
# ### Feature Importance

# %%
# Create a pd.Series of features importances
importances = pd.Series(data=rf.feature_importances_,
                        index= X_train.columns)

# Sort importances
importances_sorted = importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12,15))
importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# %%
rf_cv_scores = cross_val_score(rf, X, y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold cross-validation scores
print(rf_cv_scores)
lm_rf_score_rmse = abs(rf_cv_scores)**(1/2)
print(lm_rf_score_rmse)

print("Average 10-Fold CV Score: {}".format(np.mean(lm_rf_score_rmse)))

# %% [markdown]
# ## XGBoost Model

# %%
# Import xgboost
import xgboost as xgb

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(objective="reg:squarederror",n_estimator=10,seed=123)

# Fit the regressor to the training set
xg_reg.fit(X_train, y_train)

# %% [markdown]
# ### XGBoost Model Evaluation

# %%
# Predict the labels of the test set: preds
pred_xgb = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse_test_xgb =(MSE(y_test, pred_xgb))**(1/2)
print("RMSE: %f" % (rmse_test_xgb))


# %%
# Create a pd.Series of features importances
xgb_importances = pd.Series(data=xg_reg.feature_importances_,
                        index= X_train.columns)

# Sort importances
xgb_importances_sorted = xgb_importances.sort_values()

# Draw a horizontal barplot of importances_sorted
plt.figure(figsize=(12,15))
xgb_importances_sorted.plot(kind='barh', color='lightgreen')
plt.title('Features Importances')
plt.show()


# %%
xgb_cv_scores = cross_val_score(xg_reg, X, y, cv=10, scoring="neg_mean_squared_error")

# Print the 10-fold cross-validation scores
print(xgb_cv_scores)
lm_xgb_score_rmse = abs(xgb_cv_scores)**(1/2)
print(lm_xgb_score_rmse)

print("Average 10-Fold CV Score: {}".format(np.mean(lm_xgb_score_rmse)))

