#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
import numpy as np
import openpyxl  # Used for exporting the final cross-tabulation to an .xlsx file
import matplotlib.pyplot as plt
import japanize_matplotlib
import collections
import os

# Handling command-line arguments
import argparse
# Create a parser for command-line arguments
parser = argparse.ArgumentParser()
# Define and add the command-line arguments
parser.add_argument('arg1')  # Argument 1: Path to the working directory for the UI
parser.add_argument('arg2')  # Argument 2: Folder name to be processed by this script
# Parse the provided arguments
args = parser.parse_args()

# Define constants
S_WORKINGPATH = args.arg1
S_FOLDERNAME = args.arg2
S_PATH = S_WORKINGPATH + '\/' + S_FOLDERNAME  # Concatenate paths with '/'
os.chdir(S_PATH)
S_FILENAME_AI = "annotation.csv"
S_FILENAME_SEIKAI = "annotation.csv_new"

L_COLUMN_NAMES = ['X', 'Y', 'CELLTYPE', 'PROB', 'COLOR', 'DR']  # Column names for the input data

L_ANALYSYS_CELLTYPE = [
    'adenocarcinoma_NOS', 'lymphocyte', 'glandular_epithelium',
    'stromal_cell', 'double_mm', 'red_m',
    'green_m', 'macrophage', 'true_overcount',
    'false_overcount', 'non-cellular'
]  # Cell types to be analyzed

L_SEIKAI_ROW = L_ANALYSYS_CELLTYPE + ['All']  # Cell types used for analysis, including an 'All' category

L_AI_COLUMN = [
    'adenocarcinoma_NOS', 'lymphocyte', 'glandular_epithelium',
    'stromal_cell', 'double_mm', 'red_m',
    'green_m', 'macrophage', 'overcount',
    'non-cellular', 'All'
]  # Aggregated cell types for AI data

I_SCREENCUTOFF = 50  # Threshold for excluding cells that are outside the valid range
I_VSLIDE_WIDTH = 802  # Width of the virtual slide
I_VSLIDE_HEIGHT = 802  # Height of the virtual slide

I_FRAMESIZE = 50  # Length of one side of a grid square (in pixels)
I_TOLERABLE_ERROR_PIXELS = 40  # Radius of acceptable error for matching cells (in pixels)

# Load AI and ground truth annotation data
df_annotated = pd.read_csv(S_FILENAME_AI, encoding="shift-jis", sep='\t', header=None, names=L_COLUMN_NAMES)
df_seikai = pd.read_csv(S_FILENAME_SEIKAI, encoding="shift-jis", sep='\t', header=None, names=L_COLUMN_NAMES)

# Assign sequential IDs starting from 1 to the pins in both datasets
l_ID_seikai = list(range(1, df_seikai.shape[0] + 1))
df_seikai['ID'] = l_ID_seikai
l_ID_annotated = list(range(1, df_annotated.shape[0] + 1))
df_annotated['ID'] = l_ID_annotated

# Determine the total number of grid squares in the horizontal (X) and vertical (Y) directions
i_Frame_X = 0
i_Frame_Y = 0
i, j = 0, 0  # Initialize counters for X and Y dimensions
while i < I_VSLIDE_WIDTH:
    i += I_FRAMESIZE
    i_Frame_X += 1
i_Frame_X -= 1

while j < I_VSLIDE_HEIGHT:
    j += I_FRAMESIZE
    i_Frame_Y += 1
i_Frame_Y -= 1

print(i_Frame_X, i_Frame_Y)

# Function 0: Calculate the squared distance between two points (x, y) and (a, b)
def func_Dist_Square(x, y, a, b):
    i = (x-a)**2 + (y-b)**2
    return i

# Function 1: Determine which grid square a point (x, y) belongs to
def func_Allocation(x,y):
    glid_x=0
    glid_y=0
    if x < (I_VSLIDE_WIDTH // I_FRAMESIZE) * I_FRAMESIZE:
        glid_x = x // I_FRAMESIZE
    else:
        glid_x = (x // I_FRAMESIZE) - 1
        
    if y < (I_VSLIDE_HEIGHT // I_FRAMESIZE) * I_FRAMESIZE:
        glid_y = y // I_FRAMESIZE
    else:
        glid_y = (y // I_FRAMESIZE) - 1
    return(glid_x, glid_y)

# Function 2: Assign grid coordinates (GRID_X, GRID_Y) to each point in a DataFrame
def func_Grid(df):
    for i in range(1, df.shape[0] + 1, 1):
        x = df.loc[df['ID']==i, 'X']
        x=x.values[0]
        y = df.loc[df['ID']==i, 'Y']
        y=y.values[0]
        v_grid =func_Allocation(x,y)

        df.loc[df['ID']==i, 'GRID_X'] = v_grid[0]
        df.loc[df['ID']==i, 'GRID_Y'] = v_grid[1]

# Perform grid assignment for AI and ground truth data
func_Grid(df_annotated)
func_Grid(df_seikai)

# Function 3: For a given point in the AI dataset, return a list of ground truth cell IDs
# located within the 9 surrounding grid squares.
# This function has two steps:

# Step 3-1: Get a list of ground truth cell IDs within a specific grid (grid_x, grid_y).
def func_SingleGrid_Cell_ID(grid_x, grid_y): 
    l = df_seikai.loc[(df_seikai['GRID_X']==grid_x) & (df_seikai['GRID_Y']==grid_y), 'ID'].tolist()
    return l

# Step 3-2: For a given AI cell ID, find the IDs of ground truth cells in the surrounding grids.
def func_Area_Cell_ID(annotated_id):
    # Determine the grid coordinates (grid_x, grid_y) of the given AI cell ID
    grid_x = df_annotated.loc[df_annotated['ID'] == annotated_id, 'GRID_X'].values[0]
    grid_y = df_annotated.loc[df_annotated['ID'] == annotated_id, 'GRID_Y'].values[0]
    # Based on grid_x and grid_y, calculate the list of horizontal (l_GridArea_X) and vertical (l_GridArea_Y) grid ranges to include surrounding grids
    if grid_x == 0:  # At the left edge
        l_GridArea_X = [0, 1]
    elif grid_x == i_Frame_X - 1:  # At the right edge
        l_GridArea_X = [i_Frame_X - 2, i_Frame_X - 1]
    else:  # Cells not on the edge
        l_GridArea_X = [grid_x - 1, grid_x, grid_x + 1]

    if grid_y == 0:  # At the top edge
        l_GridArea_Y = [0, 1]
    elif grid_y == i_Frame_Y - 1:  # At the bottom edge
        l_GridArea_Y = [i_Frame_Y - 2, i_Frame_Y - 1]
    else:  # Cells not on the edge
        l_GridArea_Y = [grid_y - 1, grid_y, grid_y + 1]

    # Initialize the list to store ground truth cell IDs within the 9 surrounding grids
    l_Area_Cell_ID = []
    for i in l_GridArea_X:
        for j in l_GridArea_Y:
            l = func_SingleGrid_Cell_ID(i, j)
            l_Area_Cell_ID += l  # Append cell IDs from each grid
    return l_Area_Cell_ID


# Function 4: Find the nearest ground truth cell for a given AI cell
# Input: AI cell ID (annotated_id)
# Output: Closest ground truth cell ID and its squared distance
def func_Nearest_Seikai_ID(annotated_id):
    l = func_Area_Cell_ID(annotated_id)  # Get ground truth cell IDs in the surrounding area
    if not l:
        # If no nearby ground truth cells exist, return 0 as the match
        return 0
    else:
        # Compute squared distances between the AI cell and all nearby ground truth cells
        x = df_annotated.loc[df_annotated['ID'] == annotated_id, 'X'].values[0]
        y = df_annotated.loc[df_annotated['ID'] == annotated_id, 'Y'].values[0]
        l_Dist_Square_Area = [
            func_Dist_Square(x, y, 
                             df_seikai.loc[df_seikai['ID'] == i, 'X'].values[0], 
                             df_seikai.loc[df_seikai['ID'] == i, 'Y'].values[0]) 
            for i in l
        ]
        # Find the ground truth cell with the minimum distance
        i = l_Dist_Square_Area.index(min(l_Dist_Square_Area))
        zantei_taiou_seikai_id = l[i]
        return zantei_taiou_seikai_id, min(l_Dist_Square_Area)

## Use the defined function to add a new column ZANTEI_TAIOU_SEIKAI_ID to df_annotated
# Function 5: Define func_screencutoff to remove records outside the valid range
# This function should be called after assigning ID_Seikai.
## Identify records that should be excluded due to being out of range and store them in df_outrange
### The exclusion criteria are: { (X <= threshold or X >= width - threshold) or (Y <= threshold or Y >= height - threshold) }
def func_screencutoff(df, i_vslide_width, i_vslide_height, i_screencutoff):
    df_outrange = df.loc[
        ((df['X'] <= i_screencutoff) | (df['X'] >= (i_vslide_width - i_screencutoff))) | 
        ((df['Y'] <= i_screencutoff) | (df['Y'] >= (i_vslide_height - i_screencutoff)))
    ]
    diff_df = pd.concat([df, df_outrange]).drop_duplicates(keep=False)
    return diff_df

# Exclude records outside the defined screenable range
## Exclude non-target records, keeping only points corresponding to the cell types of interest
df_annotated = df_annotated[df_annotated['CELLTYPE'].isin(L_ANALYSYS_CELLTYPE)]
df_seikai = df_seikai.loc[df_seikai['CELLTYPE'] != 'off-target']
## Apply out-of-range processing for both the AI file and the ground truth file
df_annotated = func_screencutoff(df_annotated, I_VSLIDE_WIDTH, I_VSLIDE_HEIGHT, I_SCREENCUTOFF)
df_seikai = func_screencutoff(df_seikai, I_VSLIDE_WIDTH, I_VSLIDE_HEIGHT, I_SCREENCUTOFF)
###### Update the list to exclude removed pins from future loops
l_id_annotated = df_annotated['ID'].tolist()
l_id_seikai = df_seikai['ID'].tolist()


# Assign the nearest ground truth cell ID to each AI cell
for i in l_id_annotated:
    a = func_Nearest_Seikai_ID(i)
    if a == 0:
        # If no match is found, assign 0 to the correspondence column
        df_annotated.loc[df_annotated['ID'] == i, 'TAIOU_SEIKAI_ID'] = 0    
    else:
        # Otherwise, assign the matched ground truth cell ID and related properties
        df_annotated.loc[df_annotated['ID'] == i, 'TAIOU_SEIKAI_ID'] = a[0]
        df_annotated.loc[df_annotated['ID'] == i, 'X_SEIKAI'] = df_seikai.loc[df_seikai['ID'] == a[0], 'X'].values[0]
        df_annotated.loc[df_annotated['ID'] == i, 'Y_SEIKAI'] = df_seikai.loc[df_seikai['ID'] == a[0], 'Y'].values[0]
        df_annotated.loc[df_annotated['ID'] == i, 'DIST_SQUARE'] = a[1]

# Add a 'Checked' column to classify match quality
# Assign "OK" for distances within the tolerable error range, otherwise "Needed"
df_annotated.loc[df_annotated['DIST_SQUARE'] <= I_TOLERABLE_ERROR_PIXELS ** 2, "Checked"] = "OK"
df_annotated.loc[df_annotated['DIST_SQUARE'] > I_TOLERABLE_ERROR_PIXELS ** 2, "Checked"] = "Needed"

# Rename columns for consistency 
df_annotated = df_annotated.rename(columns={'TAIOU_SEIKAI_ID': 'ID_SEIKAI'})
df_annotated = df_annotated.rename(columns={'ID': 'ID_AI'})
df_annotated = df_annotated.rename(columns={'CELLTYPE': 'CELLTYPE_AI'})
df_seikai = df_seikai.rename(columns={'CELLTYPE': 'CELLTYPE_SEIKAI'})


#####################                                       Initial Processing

# Definition of functions

''' Variables used (only global)
serial_num_Seikai
l_serial_num_Seikai - List of ID_Seikai in the ground truth file but not in the AI file
l_lack_ID_Seikai - List of IDs for missing records in the ground truth file
l_duplicated_ID_Seikai - List of duplicate ID_Seikai within the AI file (used for overcount calculations)
df_AI
df_Seikai
df_Temp
df_Analysis
'''

######################                                           DataFrame Loading
## Load the AI file 
df_AI = df_annotated
## Load the ground truth file 
df_Seikai = df_seikai
## Assign serial numbers to the records in df_Seikai
serial_num_Seikai = pd.RangeIndex(start=1, stop=len(df_Seikai) + 1, step=1)
df_Seikai['ID_SEIKAI'] = serial_num_Seikai
l_serial_num_Seikai = serial_num_Seikai.tolist()

#####################                                         Create df_Analysis

# Processing up to merging
## Merge df_AI and df_Seikai using ID_SEIKAI as the key
df_Analysis = pd.merge(df_AI, df_Seikai[['CELLTYPE_SEIKAI', 'ID_SEIKAI']], on='ID_SEIKAI', how='left')
## Retain only the necessary columns
df_Analysis = df_Analysis[['X', 'Y', 'COLOR', 'ID_AI', 'PROB', 'CELLTYPE_AI', 'ID_SEIKAI', 'CELLTYPE_SEIKAI', 'DIST_SQUARE', 'Checked']]

## Create a list of IDs (l_lack_ID_Seikai) for records present in df_Seikai but not in df_AI
l_lack_ID_Seikai = list(set(l_serial_num_Seikai) - set(df_Analysis['ID_SEIKAI'].tolist()))
l_lack_ID_Seikai.sort()
## Extract the corresponding records from df_Seikai using the list (df_Temp)
df_Temp = df_Seikai[df_Seikai['ID_SEIKAI'].isin(l_lack_ID_Seikai)]
## Assign CELLTYPE_AI = 'non-cellular' to df_Temp
df_Temp['CELLTYPE_AI'] = 'non-cellular'
df_Temp['ID_AI'] = 0
## Reorganize the columns
df_Temp = df_Temp[['X', 'Y', 'COLOR', 'ID_AI', 'PROB', 'CELLTYPE_AI', 'ID_SEIKAI', 'CELLTYPE_SEIKAI']]
## Merge df_Temp into df_Analysis
df_Analysis = pd.concat([df_Analysis, df_Temp], axis=0)
## For records with Checked == 'Needed', set CELLTYPE_SEIKAI to 'non-cellular'
df_Analysis.loc[df_Analysis['Checked'] == 'Needed', 'CELLTYPE_SEIKAI'] = 'non-cellular'
## For records with ID_SEIKAI == 0, set CELLTYPE_SEIKAI to 'non-cellular'
df_Analysis.loc[df_Analysis['ID_SEIKAI'] == 0, 'CELLTYPE_SEIKAI'] = 'non-cellular'

# Remove records containing non-target cells from the completed df_Analysis
## Exclude records where both CELLTYPE_AI and CELLTYPE_SEIKAI are not target cells
df_Analysis = df_Analysis.loc[
    (df_Analysis['CELLTYPE_AI'].isin(L_ANALYSYS_CELLTYPE)) &
    (df_Analysis['CELLTYPE_SEIKAI'].isin(L_ANALYSYS_CELLTYPE))
]
df_Analysis.shape
## Assign CELLTYPE_AI = 'non-cellular' for l_lack_ID_Seikai
## Remove records where both CELLTYPE_AI and CELLTYPE_SEIKAI are 'non-cellular'
df_Analysis = df_Analysis[~(df_Analysis['CELLTYPE_AI'] == 'non-cellular') | ~(df_Analysis['CELLTYPE_SEIKAI'] == 'non-cellular')]

####################                                               Handle Duplicates

# After removing unnecessary records, evaluate duplicates. Retrieve a list of duplicated ID_Seikai from df_Analysis.
l_duplicated_ID_Seikai = [k for k, v in collections.Counter(df_Analysis['ID_SEIKAI'].tolist()).items() if v > 1]
l_duplicated_ID_Seikai  

# Handle overcounting of duplicate records
## Process the list of duplicated ID_Seikai
for i01 in l_duplicated_ID_Seikai:
    ### Retrieve the list of ID_AI for which ID_SEIKAI == i01
    l01 = df_Analysis.loc[df_Analysis['ID_SEIKAI'] == i01, 'ID_AI'].tolist()
    l02 = []  ## Create a list of corresponding DIST_SQUARE values for l01 in order
    for j1 in l01:  ## Append DIST_SQUARE values to l02
        l02.append(df_Analysis.loc[df_Analysis['ID_AI'] == j1, 'DIST_SQUARE'].values[0])

    j2 = l02.index(min(l02))  # Index of the smallest DIST_SQUARE in l02
    j3 = l01[j2]  ## ID_AI corresponding to the smallest DIST_SQUARE
    l01.remove(j3)  ## Remove j3 from l01 to create a list of overcounted pins

    ### Separate l01 into correctly matched (l01_true) and incorrectly matched (l01_false)
    l01_true = [i for i in l01 if df_Analysis.loc[df_Analysis['ID_AI'] == i, "CELLTYPE_AI"].values[0] == df_Analysis.loc[df_Analysis['ID_AI'] == i, "CELLTYPE_SEIKAI"].values[0]]
    l01_false = [i for i in l01 if df_Analysis.loc[df_Analysis['ID_AI'] == i, "CELLTYPE_AI"].values[0] != df_Analysis.loc[df_Analysis['ID_AI'] == i, "CELLTYPE_SEIKAI"].values[0]]
    ### Assign true_overcount or false_overcount based on proximity and correctness
    df_Analysis.loc[(df_Analysis["Checked"] == "OK") & df_Analysis['ID_AI'].isin(l01_true), "CELLTYPE_SEIKAI"] = 'true_overcount'
    df_Analysis.loc[(df_Analysis["Checked"] == "OK") & df_Analysis['ID_AI'].isin(l01_false), "CELLTYPE_SEIKAI"] = 'false_overcount'

# Change colors for verification
df_Analysis["COLOR"] = "green"  # Default is green
df_Analysis.loc[df_Analysis["CELLTYPE_SEIKAI"] == "true_overcount", "COLOR"] = "yellow"  # True matches are yellow
df_Analysis.loc[df_Analysis["CELLTYPE_SEIKAI"] == "false_overcount", "COLOR"] = "orange"  # False matches are orange
df_Analysis.loc[df_Analysis["CELLTYPE_SEIKAI"] == "non-cellular", "COLOR"] = "blue"  # Non-cellular is blue

####################                                          Cross-Tabulation and Result Display
# Perform cross-tabulation
df_crosstab = pd.crosstab(df_Analysis['CELLTYPE_SEIKAI'], df_Analysis['CELLTYPE_AI'], margins=True)

'''
STEP1: Identify missing columns
STEP2: Add missing columns
STEP3: Identify missing rows
STEP4: Add missing rows
STEP5: Reorder the rows and columns
STEP6: Export the results
'''

## STEP1
### Retrieve a list of missing columns and store it in l_lack_columns
l_lack_columns = list(set(L_AI_COLUMN) - set(df_crosstab.columns.tolist()))
l_lack_columns

## STEP2
# Add missing columns with default values set to 0
for i in l_lack_columns:
    df_crosstab[i] = 0
df_crosstab

## STEP3
### Retrieve a list of missing rows and store it in l_lack_rows
l_lack_rows = list(set(L_SEIKAI_ROW) - set(df_crosstab.index.tolist()))
l_lack_rows

## STEP4: Adding rows is not as straightforward as adding columns. Create a new DataFrame initialized with 0 values for missing rows and merge it.
### Create an empty DataFrame with specified column names and row indices
l = df_crosstab.columns.tolist()
df_lack_rows = pd.DataFrame(columns=l, index=l_lack_rows)
### Initialize all column values to 0
for i in l:
    df_lack_rows[i] = 0
### Merge the new rows into df_crosstab
df_result = pd.concat([df_crosstab, df_lack_rows], axis=0)
df_result

## STEP5: Reorder the rows and columns
df_result = df_result.reindex(index=L_SEIKAI_ROW, columns=L_AI_COLUMN)

## STEP6: Export the results to an Excel file
os.chdir('../')  # Change working directory back to the UI directory
j = "Result" + str(S_FOLDERNAME[-6:]) + ".xlsx"  # Generate the file name
df_result.to_excel(j)

''' Portions commented out in ver5
## 2: For verification
df_Analysis.to_csv("Temporary_df.csv", index=False)
## 3: For use with Cu-Cyto
df_Analysis = df_Analysis[['X', 'Y', 'CELLTYPE_SEIKAI', 'PROB', 'COLOR']]
df_Analysis.to_csv('annotation.csv', sep='\t', header=False, index=False)
'''
print("Processing for folder " + S_FOLDERNAME + " is complete!")
