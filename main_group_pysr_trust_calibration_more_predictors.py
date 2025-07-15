#!/usr/bin/env python3
# External packages
from pickle import TRUE
import numpy as np
from pysr import PySRRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sympy
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import itertools

import warnings
# Filter out Pandas warnings
warnings.filterwarnings("ignore")

from pathlib import Path

results_path_split_groups = Path("results") / "PySR" / "split_groups"
results_path_split_groups.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

results_path_more_predictors = Path("results") / "PySR" / "more_predictors"
results_path_more_predictors.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists

#### ATTENTION: PySR only supports numeric data, categorical data is NOT supported
# requires encoding:
#One-Hot Encoding: Preferred when the categorical variables have no intrinsic order and the model can benefit from treating each category as a separate feature.
#Label Encoding: Useful for ordinal data where the categories have a natural order (e.g., 'low', 'medium', 'high').


# read data

# Specify the file path
file_path = Path("data") / "all_combined_prepared_with_demographics.xlsx"
#file_path_removed_DEI = Path("data") / "all_combined_prepared_removed_REI.xlsx"

# Specify the sheet name (optional)
sheet_name = "Sheet1"

# stays the same
model = PySRRegressor(
    niterations=500,  # < Increase me for better results
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["sin", "square", "tan", "cos", "cube", "tanh", 
                     "sqrt", "abs", "log", "exp", "cos2(x)=cos(x)^2",
                       "quart(x) = x^4", "inv(x) = 1/x"
                       # ^ Custom operator (julia syntax)
                       ], 
    #extra_sympy_mappings={"inv": lambda x: 1 / x},
    extra_sympy_mappings={
         "cos2": lambda x: sympy.cos(x)**2,
         "inv": lambda x: 1 / x, 
         "quart": lambda x: x**4},
    # ^ Define operator for SymPy as well
    #loss="loss(prediction, target) = (prediction - target)^2",
    # ^ Custom loss function (julia syntax)
     constraints={
        "^": (-1, 1),
    },
    ncyclesperiteration=2500,
    maxsize=10,
    precision=32,
    turbo=True,
)


# List of file paths
file_paths = [file_path]

# Dictionary to store DataFrames read from each Excel file
dfs = {}

# Loop through the list of file paths to read each Excel file into a DataFrame
for path in file_paths:
    
    # Extract the name without the '.xlsx' extension using pathlib
    name_without_extension = path.stem


    # Read the Excel file into a DataFrame
    df = pd.read_excel(path, sheet_name=sheet_name)

    df.dropna(inplace=True)

    # Display the DataFrame
    print(df.head())


    # Get unique values for "INTRODUCTION" and "SCENARIO" columns
    unique_introductions = df['INTRODUCTION'].unique()
    unique_scenarios = df['SCENARIO'].unique()
    unique_prolifics = df['ProlificID'].unique()

    # Add a new column to df that combines 'INTRODUCTION' and 'SCENARIO'
    df['intro_scenario_combo'] = df['INTRODUCTION'].astype(str) + "_" + df['SCENARIO'].astype(str)


    print("df")
    print(df.shape)

    # Create a dictionary to store the number of times each value appears in the 'trust' column, per combination of ProlificID, INTRODUCTION, and SCENARIO
    trust_counts = {}
    for id, introduction, scenario in df[['ProlificID', 'INTRODUCTION', 'SCENARIO']].values:
        trust_counts[(id, introduction, scenario)] = df[(df['ProlificID'] == id) & (df['INTRODUCTION'] == introduction) & (df['SCENARIO'] == scenario)]['trust'].value_counts()

    # Create a list of tuples to store the combinations of values that appear 8 or more times
    #combinations = []
    #for key, value in trust_counts.items():
    #    one_was_eight = False
    #    for value1, value2 in value.items():
    #        if value2 >= 14:
    #            combinations.append((key))
    #        elif value2 >= 7:
    #            if one_was_eight:
    #                combinations.append((key))
    #            else:
    #                one_was_eight = True



    # Initialize an empty list to store the combinations
    combinations = []

    # Initialize a variable to store the last value2 that was >= 7 for each key
    last_value2_dict = {}

    # Assume trust_counts is defined; iterate through its items
    for key, value in trust_counts.items():
        one_was_eight = False
        
        # Iterate through the value dictionary for each key
        for value1, value2 in value.items():
            
            # If value2 is >= 14, append the key to combinations
            if value2 >= 14:
                combinations.append(key)
            
            # If value2 is >= 7, check additional conditions
            elif value2 >= 7:
                
                # If one_was_eight is True and the last value2 for this key is within +/- 1 of the current value2
                if one_was_eight and abs(last_value2_dict.get(key, 0) - value2) <= 1:
                    combinations.append(key)
                
                # Else, set one_was_eight to True and update the last value2 for this key
                else:
                    one_was_eight = True
                    last_value2_dict[key] = value2


    #print(trust_counts)
    #print(combinations)
    # Create a new DataFrame to store the data for the splitted data
    all_equal_df = pd.DataFrame()

    # Iterate over the combinations
    for combination in combinations:
        # Filter the data for the combination
        filtered_df = df[(df['ProlificID'] == combination[0]) & (df['INTRODUCTION'] == combination[1]) & (df['SCENARIO'] == combination[2])]

        # Add the filtered data to the splitted DataFrame
        all_equal_df = pd.concat([all_equal_df, filtered_df])


    # Create a new DataFrame to store the data for the splitted data
    #other_rows_df = pd.DataFrame()

    # Exclude the filtered data from the original DataFrame
    print(df[:5])
    other_rows_df = df[~df.isin(all_equal_df)].dropna()

    print("other_rows_df")
    #print(other_rows_df.shape)
    #print(other_rows_df[:5])




    x_values = other_rows_df['mIoU'].dropna().to_numpy()
    x_values = x_values.reshape(-1, 1)

    # New dimension based on the 'SCENARIO' column
    # non-categorical
    x_scenario = other_rows_df['SCENARIO'].dropna().to_numpy()
    x_scenario = x_scenario.reshape(-1, 1)

    # # New dimension based on the 'GENDER' column
    x_gender = other_rows_df['Gender'].dropna().to_numpy()
    x_gender = x_gender.reshape(-1, 1)

    #     # New dimension based on the 'AGE' column
    x_age = other_rows_df['Age'].dropna().to_numpy()
    x_age = x_age.reshape(-1, 1)

    #     # New dimension based on the 'Education' column
    x_education = other_rows_df['Education'].dropna().to_numpy()
    x_education = x_education.reshape(-1, 1)

    #     # New dimension based on the 'Job' column
    x_job = other_rows_df['Job'].dropna().to_numpy()
    x_job = x_job.reshape(-1, 1)

    #     # New dimension based on the 'License' column
    x_license = other_rows_df['License'].dropna().to_numpy()
    x_license = x_license.reshape(-1, 1)

    #     # New dimension based on the 'DrivingFrequency' column
    x_drivingfreq = other_rows_df['DrivingFrequency'].dropna().to_numpy()
    x_drivingfreq = x_drivingfreq.reshape(-1, 1)

    #     # New dimension based on the 'Distance' column
    x_distance = other_rows_df['Distance'].dropna().to_numpy()
    x_distance = x_distance.reshape(-1, 1)

    # New dimension based on the 'INTRODUCTION' column
    x_intro = other_rows_df['INTRODUCTION'].dropna().to_numpy()
    x_intro = x_intro.reshape(-1, 1)






    # One-Hot Encoding for categorical variables
    one_hot_encoder = OneHotEncoder(sparse=False)

    categorical_features = ['SCENARIO', 'Gender', 'Education', 'Job', 'INTRODUCTION']
    x_categorical = one_hot_encoder.fit_transform(other_rows_df[categorical_features])

    # Label Encoding for ordinal variables
    label_encoder = LabelEncoder()

    ordinal_features = ['License', 'DrivingFrequency', 'Distance']
    x_license = label_encoder.fit_transform(other_rows_df['License'].dropna()).reshape(-1, 1)
    x_drivingfreq = label_encoder.fit_transform(other_rows_df['DrivingFrequency'].dropna()).reshape(-1, 1)
    x_distance = label_encoder.fit_transform(other_rows_df['Distance'].dropna()).reshape(-1, 1)

    # Combining all features
    x_values_extended = np.hstack((x_values, x_age, x_categorical, x_license, x_drivingfreq, x_distance))

   # Adding new dimension to x_values
    x_values_extended = np.hstack([x_values, x_age]) # x_scenario,  x_intro, x_gender, x_job,  x_license, x_drivingfreq, x_distance, x_education


 
    y_values = other_rows_df[['trust']].dropna()

    print("x_values_extended.shape")
    print(x_values_extended.shape)

    # Initialize PySRRegressor and fit the model
    # done before
    #model = PySRRegressor()
    model.fit(x_values_extended, y_values)

    print("SYMPY")
    print(model.sympy())

    print("")
    print("")
    print("LATEX")
    print(model.latex())
    print(model.latex_table())
    print("")
    print("")

    file_path = results_path_split_groups / f'model_info_other_rows_df_stacked_MULTIPLE_{name_without_extension}.txt'
    with file_path.open('w') as f:
        f.write("SYMPY\n")
        f.write(str(model.sympy()))
        f.write("\n\nLATEX\n")
        f.write(str(model.latex()))
        f.write("\n\nLATEX TABLE\n")
        f.write(str(model.latex_table()))

    # Plotting logic
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.5)

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.scatterplot(x=x_values.ravel(), y=y_values['trust'], hue=other_rows_df['intro_scenario_combo'], palette='viridis', alpha=0.3, s=50, edgecolor=None)
    #sns.scatterplot(x=x_values.ravel(), y=y_values['trust'], color='grey', alpha=0.5, s=50, edgecolor=None)
    sns.lineplot(x=x_values.ravel(), y=model.predict(x_values), color='black', lw=2)

    # Move the legend outside the plot
    #ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_xlabel('mIoU')
    ax.set_ylabel('Trust')
    ax.set_title(f'Visualization of the Equation')

    ax.set_ylim(1, 5) 

    sns.despine()

    file_path = results_path_more_predictors / f'relationship_pysr_other_rows_df_stacked_MULTIPLE_{name_without_extension}.png'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)
