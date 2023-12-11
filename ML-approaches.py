#!/usr/bin/env/python
# External packages
from pickle import TRUE
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colormaps

from mpl_toolkits.mplot3d import Axes3D
import sympy as sympy
import pandas as pd
import itertools
import shap


import sklearn
from sklearn import metrics
from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt


from sklearn.model_selection import train_test_split

from sklearn.preprocessing import OneHotEncoder

from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer, mean_absolute_error

# read data

# Specify the file path
file_path = "all_combined_prepared_with_demographics.xlsx"

# Specify the sheet name (optional)
sheet_name = "Sheet1"

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path, sheet_name=sheet_name)

df.dropna(inplace=True)

# Display the DataFrame
#print(df.head())


# Gender
df['Gender'] = df['Gender'].replace({
    "A1": "F",
    "A2": "M",
    "A3": "non-binary",
    "A4": "Prefer not to tell"
})

# Education
df['Education'] = df['Education'].replace({
    "A1": "Secondary School",
    "A2": "Middle School",
    "A3": "High School",
    "A4": "College",
    "A5": "Vocational training"
})

# Job
df['Job'] = df['Job'].replace({
    "A1": "Student (school)",
    "A2": "Student (college)",
    "A3": "Employee",
    "A4": "Self-employed",
    "A5": "Jobseeker",
    "A6": "Other"
})

# Driving frequency
df['DrivingFrequency'] = df['DrivingFrequency'].replace({
    "A1": "Daily",
    "A2": "On working days",
    "A3": "3-4 times a week",
    "A4": "1 time a week",
    "A5": "1-3 times a month",
    "A6": "less than 1 time a month"
})

# Distance
df['Distance'] = df['Distance'].replace({
    "A1": "less than 7.000km",
    "A2": "7.000 - 14.999km",
    "A3": "15.000 - 24.999km",
    "A4": "25.000 - 32.999km",
    "A5": "33.000 or more km"
})



### ONCE FOR ALL DATA

x_values = df['mIoU'].to_numpy()
x_values = x_values.reshape(-1, 1)

y_values = df[['trust']].dropna()


x_values_flat = x_values.ravel()

x_values = df['mIoU'].dropna().to_numpy()
x_values = x_values.reshape(-1, 1)

# New dimension based on the 'SCENARIO' column
x_scenario = df['SCENARIO'].dropna().to_numpy()
x_scenario = x_scenario.reshape(-1, 1)

# # New dimension based on the 'GENDER' column
x_gender = df['Gender'].dropna().to_numpy()
x_gender = x_gender.reshape(-1, 1)

#     # New dimension based on the 'AGE' column
x_age = df['Age'].dropna().to_numpy()
x_age = x_age.reshape(-1, 1)

#     # New dimension based on the 'Education' column
x_education = df['Education'].dropna().to_numpy()
x_education = x_education.reshape(-1, 1)

#     # New dimension based on the 'Job' column
x_job = df['Job'].dropna().to_numpy()
x_job = x_job.reshape(-1, 1)

#     # New dimension based on the 'License' column
x_license = df['License'].dropna().to_numpy()
x_license = x_license.reshape(-1, 1)

#     # New dimension based on the 'DrivingFrequency' column
x_drivingfreq = df['DrivingFrequency'].dropna().to_numpy()
x_drivingfreq = x_drivingfreq.reshape(-1, 1)

#     # New dimension based on the 'Distance' column
x_distance = df['Distance'].dropna().to_numpy()
x_distance = x_distance.reshape(-1, 1)

# New dimension based on the 'INTRODUCTION' column
x_intro = df['INTRODUCTION'].dropna().to_numpy()
x_intro = x_intro.reshape(-1, 1)

# Adding new dimension to x_values
x_values_extended = np.hstack([x_values, x_scenario, x_intro, x_gender, x_age, x_education, x_job, x_license, x_drivingfreq, x_distance])




print("################### DATA ##############")
print(x_values.shape)
print(y_values.shape)

print("#### x values ####")
print(x_values[:5])
print("#### y values ####")
print(y_values[:5])


print("#######################################")

# db = DBSCAN(eps=0.3, min_samples=10).fit(y_values)
# labels = db.labels_



# Prepare and scale features
features = df[['trust']]
features_scaled = StandardScaler().fit_transform(features)

# DBSCAN clustering
db = DBSCAN(eps=0.3, min_samples=10).fit(features_scaled)
labels = db.labels_



# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)


unique_labels = set(labels)
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True

colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
# Plotting
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = [0, 0, 0, 1]  # Black for noise
    
    class_member_mask = (labels == k)
    
    # Core samples
    xy = x_values[class_member_mask & core_samples_mask].ravel()
    plt.scatter(
        xy,
        y_values[class_member_mask & core_samples_mask].to_numpy().ravel(),
        c=[tuple(col)],
        edgecolor="k",
        s=60,
    )
    
    # Non-core samples
    xy = x_values[class_member_mask & ~core_samples_mask].ravel()
    plt.scatter(
        xy,
        y_values[class_member_mask & ~core_samples_mask].to_numpy().ravel(),
        c=[tuple(col)],
        edgecolor="k",
        s=30,
    )

plt.title(f"Estimated number of clusters from DBSCAN: {n_clusters_}")
#plt.show()

#plt.savefig(f'dbscan_clusters_{n_clusters_}.png', bbox_inches='tight', pad_inches=0)


# knn_graph = kneighbors_graph(features, 20, include_self=False)

# for connectivity in (None, knn_graph):
#     for n_clusters in (3, 5, 9, 20):
#         plt.figure(figsize=(10, 4))
#         for index, linkage in enumerate(("average", "complete", "ward", "single")):
#             plt.subplot(1, 4, index + 1)
#             model = AgglomerativeClustering(
#                 linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
#             )
#             t0 = time.time()
#             model.fit(features)
#             elapsed_time = time.time() - t0
#             plt.scatter(x_values, y_values, c=model.labels_, cmap=plt.cm.nipy_spectral)
#             plt.title(
#                 "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
#                 fontdict=dict(verticalalignment="top"),
#             )
#             plt.axis("equal")
#             plt.axis("off")

#             plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
#             plt.suptitle(
#                 "n_cluster=%i, connectivity=%r"
#                 % (n_clusters, connectivity is not None),
#                 size=17,
#             )

#             plt.savefig(f'Agglomerative_clustering_{linkage}_nr_cluster_{n_clusters}.png', bbox_inches='tight', pad_inches=0)
#plt.show()







################### Multiple Features


df_original_catboost = df.copy(deep=True)
df_original_xgboost = df.copy(deep=True)
df_original = df.copy(deep=True)


# Identify numerical and categorical features
#numerical_features = ['mIoU', 'License']
#categorical_features = ['SCENARIO', 'INTRODUCTION'] #, 'Gender', 'Age', 'Education', 'Job', 'DrivingFrequency', 'Distance']

print(df.columns)

# Initialize OneHotEncoder
encoder = OneHotEncoder(sparse_output=False, drop='first')

# Separate numerical and categorical features
numerical_features = ['mIoU', 'License', 'Age']
categorical_features = ['SCENARIO', 'INTRODUCTION', 'Gender',  'Education', 'Job', 'DrivingFrequency', 'Distance']

# Fit and transform categorical data
categorical_data = df[categorical_features]
one_hot_encoded = encoder.fit_transform(categorical_data)

# Create DataFrame from encoded data
one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(categorical_features))

# Concatenate with original data
df.drop(categorical_features, axis=1, inplace=True)
df = pd.concat([df, one_hot_df], axis=1)

print("Data types after one-hot encoding:")
print(df.dtypes)

# Prepare Features and Target variable
#X = df.drop(['trust'], axis=1)
X = df[numerical_features + list(one_hot_df.columns)]
y = df['trust']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
forest = RandomForestRegressor(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)

# Evaluate the model
y_pred = forest.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)


print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")


#feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
feature_names = X_train.columns.tolist()
#forest = RandomForestClassifier(random_state=0)
#forest.fit(x_values, y_values)


#start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)


# TODO permutation, look at in thermal comfot or VEmotion
#permutation_importances = forest.

#elapsed_time = time.time() - start_time

#print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

# fig, ax = plt.subplots()
# forest_importances.plot.bar(yerr=std, ax=ax)
# ax.set_title("Feature importances using MDI")
# ax.set_ylabel("Mean decrease in impurity")
# fig.tight_layout()
# plt.show()



# Apply a Seaborn style
sns.set(style="whitegrid")

# Prepare the data
forest_importances = pd.Series(importances, index=feature_names)


# Create the plot
plt.figure(figsize=(12, 8))  # Optional: Set figure size
sns.barplot(x=forest_importances.index, y=forest_importances.values, yerr=std, palette="muted")

# Add labels and title
plt.xlabel('')
plt.ylabel('Importance')
plt.title('Feature Importances using Mean Decrease in Impurity')

# Add Metrics to the plot
metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}"
plt.text(0.75, 0.6, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))


# Optional: Rotate x labels for better visibility
plt.xticks(rotation=90)

sns.despine()

# Show the plot
plt.tight_layout()
#plt.show()



plt.savefig(f'feature_importance_random_classifier.png', bbox_inches='tight', pad_inches=0)




########## PERMUTATION IMPORTANCE ##################



# Create a scorer from the performance metric
mae_scorer = make_scorer(mean_absolute_error, greater_is_better=False)

# Calculate permutation importance using MAE as the scoring metric
perm_importance = permutation_importance(forest, X_test, y_test, n_repeats=30,
                                         random_state=42, scoring=mae_scorer)


# Evaluate the model
y_pred = forest.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)


print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Extract importances and convert to a suitable form
importances = perm_importance.importances_mean
std = perm_importance.importances_std
features = X_test.columns

# Convert to DataFrame for Seaborn
importance_data = pd.DataFrame({
    'Feature': features,
    'Importance': importances,
    'Std': std
})

# Sort by importance
importance_data = importance_data.sort_values(by='Importance', ascending=False)

# Start plotting
plt.figure(figsize=(12, 8))  # Set figure size

# Create barplot using Seaborn
sns.barplot(x='Importance', y='Feature', data=importance_data, palette="muted", xerr=importance_data['Std'])

# Add labels and title
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Permutation Importances using Mean Decrease in Impurity')

# Optional: Rotate x labels for better visibility
plt.xticks(rotation=90)

sns.despine()  # Removes the top and right border of the plot

# Optional: You can include additional metrics as text in the plot if you have these values
metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}"
plt.text(0.75, 0.6, metrics_text, transform=plt.gca().transAxes, 
          fontsize=12, verticalalignment='bottom', 
          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

plt.tight_layout()  # Adjust the plot to ensure everything fits without overlapping
# plt.show()  # Display the plot

# Save the plot as a file
plt.savefig('perm_importances_random_regressor.png', bbox_inches='tight', pad_inches=0)





explainer = shap.TreeExplainer(forest)
shap_values = explainer.shap_values(X_test)

# Increase the figure size for better readability
plt.figure(figsize=(10, 8))
# Generate the SHAP summary plot with a specified color palette
shap.summary_plot(shap_values, X_test, plot_type="bar", cmap=plt.get_cmap('coolwarm'), show=False)

# Adjust layout to fit and prevent label cut-off
plt.tight_layout()

# Save the plot in high resolution
plt.savefig('enhanced_shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.show()  # Display the plot}










########## catboost IMPORTANCE ##################


# try other version that can handle categorical data natively
from catboost import CatBoostRegressor
numerical_features = ['mIoU', 'License', 'Age']
categorical_features = ['SCENARIO', 'INTRODUCTION', 'Gender', 'Education', 'Job', 'DrivingFrequency', 'Distance']
target_column = 'trust'








# Prepare Features and Target variable
X = df_original_catboost[numerical_features + categorical_features]
y = df_original_catboost[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor
cat_features_indices = [X_train.columns.get_loc(c) for c in categorical_features]
model = CatBoostRegressor(cat_features=cat_features_indices, random_state=42)

# Train the model
model.fit(X_train, y_train, verbose=False)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

# Output the metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Get feature importances
importances = model.get_feature_importance()

# Map feature importances to feature names
feature_names = X_train.columns.tolist()
forest_importances = pd.Series(importances, index=feature_names)

# Apply a Seaborn style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(12, 8))
sns.barplot(x=forest_importances.index, y=forest_importances.values, palette="muted")

# Add labels and title
plt.xlabel('')
plt.ylabel('Importance')
plt.title('Feature Importances using catboost')

# Add Metrics to the plot
metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}"
plt.text(0.75, 0.6, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

# Optional: Rotate x labels for better visibility
plt.xticks(rotation=90)

sns.despine()

# Show the plot
plt.tight_layout()
plt.savefig('feature_importance_catboost.png', bbox_inches='tight', pad_inches=0)










########## XGBoost IMPORTANCE ##################


# try it with XGBoost
import xgboost as xgb

# Convert categorical columns to 'category' type
for feature in categorical_features:
    df_original_xgboost[feature] = df_original_xgboost[feature].astype('category')

# Prepare Features and Target variable
X = df_original_xgboost[numerical_features + categorical_features]
y = df_original_xgboost[target_column]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize XGBoost Regressor
model = xgb.XGBRegressor(random_state=42, enable_categorical=True)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = sqrt(mse)

# Output the metrics
print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Get feature importances
importances = model.feature_importances_

# Map feature importances to feature names
feature_names = X_train.columns.tolist()
forest_importances = pd.Series(importances, index=feature_names)


# Apply a Seaborn style
sns.set(style="whitegrid")

# Create the plot
plt.figure(figsize=(12, 8))
sns.barplot(x=forest_importances.index, y=forest_importances.values, palette="muted")

# Add labels and title
plt.xlabel('')
plt.ylabel('Importance')
plt.title('Feature Importances using XGBoost')

# Add Metrics to the plot
metrics_text = f"MAE: {mae:.4f}\nMSE: {mse:.4f}\nRMSE: {rmse:.4f}"
plt.text(0.75, 0.6, metrics_text, transform=plt.gca().transAxes, fontsize=12, verticalalignment='bottom', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black'))

# Optional: Rotate x labels for better visibility
plt.xticks(rotation=90)

sns.despine()

# Show the plot
plt.tight_layout()
plt.savefig('feature_importance_xgboost.png', bbox_inches='tight', pad_inches=0)