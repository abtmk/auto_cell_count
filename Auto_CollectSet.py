#!/usr/bin/env python
# coding: utf-8

# 0. Import necessary libraries
import os
import pandas as pd
import numpy as np
import subprocess

# 1. Define variables, constants, and functions
s_homedirpath = "your directory path"  # Path of the directory where this script is located
S_SCRIPTNAME = "your directory path/Auto_Seidokanri_ROI.py"
S_UI_NAME = "your directory path/Auto_Seidokanri_AI.py"
# s_homedirpath = os.getcwd()  # Path of the directory where this script is located
# s_currentdirpath = os.getcwd()  # Path of the current working directory

# Function to get a list of child directories within the current directory
def func_getchilddir(s_currentdirpath):    
    l_dir = [f for f in os.listdir(s_currentdirpath) if os.path.isdir(os.path.join(s_currentdirpath, f))]
    return l_dir

# 2. First, retrieve a list of directory names within the current directory
l_set_dir = [f for f in os.listdir(s_homedirpath) if os.path.isdir(os.path.join(s_homedirpath, f))]
# Retain only directories that start with the string "Set"
l_set_dir = list(filter(lambda x: x.startswith("Set"), l_set_dir))
l_set_dir  

# 3. Move into Set1 and process
for i in l_set_dir:  # Loop through each Set
    s_currentdirpath = s_homedirpath + '\\' + i  # Path of the Set-level directory
    os.chdir(s_currentdirpath)
    # Display a list of directories within it
    l_ai_dir = func_getchilddir(s_currentdirpath)
    # Retain only directories that start with the string "AI"
    l_ai_dir = list(filter(lambda x: x.startswith("AI"), l_ai_dir))
    for j in l_ai_dir:  # Loop through each AI
        # Create the full path for this folder
        k = s_currentdirpath + '\\' + j  # Path of the AI-level directory
        # Adjust backslashes for Windows compatibility
        l = k.replace("\\", "/")
        os.chdir(l)  # Move to the AI-level directory
        l_roi_dir = func_getchilddir(l)  # Retrieve a list of ROI directory names
        # print(l_roi_dir)
        
        for m in l_roi_dir:  # Loop through each ROI
            n = ["python", S_SCRIPTNAME, l, m]
            subprocess.call(n)
        # Perform aggregation tasks and output for each AI folder
        # Output the sum of values from Result~~.xlsx files in the folder

        # 1. Identify Excel files that start with "Result"
        os.chdir(l)
        l_files = [f for f in os.listdir(l) if f.startswith("Result") if f.endswith(".xlsx")]
        # 2. Create a list (l_df) of DataFrames for each Excel file
        l_df = []
        for f in l_files:
            df = pd.read_excel(f, index_col=0)
            l_df.append(df)
        # 3. Sum the DataFrames and save the result as a new Excel file
        df_sum = sum(l_df)
        df_sum
        j = "Sum_" + str(l[-3:]) + ".xlsx"  # Create the file name
        df_sum.to_excel(j)
