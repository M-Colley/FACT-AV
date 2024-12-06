#!/usr/bin/env/python
# External packages
import numpy as np
from pysr import PySRRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import sympy as sympy
import pandas as pd
import itertools

import warnings
# Filter out Pandas warnings
warnings.filterwarnings("ignore")

from pathlib import Path

results_path_personalized = Path("results") / "PySR" / "personalized_plots"
results_path_personalized.mkdir(parents=True, exist_ok=True)  # Ensure the folder exists


# read data
# Specify the file path
file_path = Path("data") / "all_combined_prepared.xlsx"
file_path_removed_DEI = Path("data") / "all_combined_prepared_removed_REI.xlsx"

# Specify the sheet name (optional)
sheet_name = "Sheet1"

# Your custom function
def custom_function(df, id, name_without_extension):
    print(f"Working with ProlificID: {id}")
    
    # Filter data based on the combination
    filtered_df = df[(df['ProlificID'] == id)]
    
    # Extract x and y values
    x_values = filtered_df['mIoU'].dropna().to_numpy()
    x_values = x_values.reshape(-1, 1)
    
    y_values = filtered_df[['trust']].dropna()

    # Initialize PySRRegressor and fit the model
    # done before
    #model = PySRRegressor()
    model.fit(x_values, y_values)

    print("SYMPY")
    print(model.sympy())

    print("")
    print("")
    print("LATEX")
    print(model.latex())
    print(model.latex_table())
    print("")
    print("")

    # Create or open a text file for writing
    file_path = results_path_personalized / f"model_info_{id}.txt"
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

    #sns.scatterplot(x=x_values.ravel(), y=y_values['trust'].values, hue=df['intro_scenario_combo'], palette='viridis', alpha=0.3, s=50, edgecolor=None)
    sns.scatterplot(x=x_values.ravel(), y=y_values['trust'], color='grey', alpha=0.5, s=50, edgecolor=None)
    sns.lineplot(x=x_values.ravel(), y=model.predict(x_values), color='green', lw=2)
    
    ax.set_xlabel('mIoU')
    ax.set_ylabel('Trust')
    ax.set_title(f'Visualization of the Equation')
    

    # Enforce y-axis to range from 1 to 5
    ax.set_ylim(1, 6) 

    sns.despine()
    file_path = results_path_personalized / f'personalized_plots/relationship_pysr_{id}_{name_without_extension}.png'
    plt.savefig(file_path, bbox_inches='tight', pad_inches=0)


# look at https://de.wikipedia.org/wiki/Polynominterpolation

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
file_paths = [file_path, file_path_removed_DEI]

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

    # Generate all combinations of unique introductions and scenarios
    all_combinations = itertools.product(unique_introductions, unique_scenarios)

    # Loop through all combinations and call your function
    for id in unique_prolifics:
        custom_function(df, id, name_without_extension)