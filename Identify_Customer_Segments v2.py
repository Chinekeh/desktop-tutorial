#!/usr/bin/env python
# coding: utf-8

# ### Prerequsite - Upgrade Scikit Learn
# The current workspace has scikit-learn v0.19.1 installed. However, you can upgrade scikit-learn to 0.24.x. and use this [OneHotEncoder](https://scikit-learn.org/0.21/modules/generated/sklearn.preprocessing.OneHotEncoder.html) library. 
# 

# In[1]:


import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[2]:


import os
os.environ['PATH'] = f"{os.environ['PATH']}:/root/.local/bin"

Restart the Kernel. 
# In[3]:


get_ipython().system('python -m pip install --upgrade scikit-learn')
import sklearn
print('The scikit-learn version is {}.'.format(sklearn.__version__))


# In[4]:


# Now you can import and use OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
# your code goes here


# In[5]:


# Similarly, should you need any other package, they can install it as:
get_ipython().system("python -m pip install 'tensorflow-tensorboard<0.2.0,>=0.1.0'")


# # Project: Identify Customer Segments
# 
# In this project, you will apply unsupervised learning techniques to identify segments of the population that form the core customer base for a mail-order sales company in Germany. These segments can then be used to direct marketing campaigns towards audiences that will have the highest expected rate of returns. The data that you will use has been provided by our partners at Bertelsmann Arvato Analytics, and represents a real-life data science task.
# 
# This notebook will help you complete this task by providing a framework within which you will perform your analysis steps. In each step of the project, you will see some text describing the subtask that you will perform, followed by one or more code cells for you to complete your work. **Feel free to add additional code and markdown cells as you go along so that you can explore everything in precise chunks.** The code cells provided in the base template will outline only the major tasks, and will usually not be enough to cover all of the minor tasks that comprise it.
# 
# It should be noted that while there will be precise guidelines on how you should handle certain tasks in the project, there will also be places where an exact specification is not provided. **There will be times in the project where you will need to make and justify your own decisions on how to treat the data.** These are places where there may not be only one way to handle the data. In real-life tasks, there may be many valid ways to approach an analysis task. One of the most important things you can do is clearly document your approach so that other scientists can understand the decisions you've made.
# 
# At the end of most sections, there will be a Markdown cell labeled **Discussion**. In these cells, you will report your findings for the completed section, as well as document the decisions that you made in your approach to each subtask. **Your project will be evaluated not just on the code used to complete the tasks outlined, but also your communication about your observations and conclusions at each stage.**

# In[6]:


# import libraries here; add more as necessary
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast 
from sklearn.impute import SimpleImputer


# magic word for producing visualizations in notebook
get_ipython().run_line_magic('matplotlib', 'inline')

'''
Import note: The classroom currently uses sklearn version 0.19.
If you need to use an imputer, it is available in sklearn.preprocessing.Imputer,
instead of sklearn.impute as in newer versions of sklearn.
'''
pd.options.display.max_rows = None
pd.options.display.max_columns = None


# ### Step 0: Load the Data
# 
# There are four files associated with this project (not including this one):
# 
# - `Udacity_AZDIAS_Subset.csv`: Demographics data for the general population of Germany; 891211 persons (rows) x 85 features (columns).
# - `Udacity_CUSTOMERS_Subset.csv`: Demographics data for customers of a mail-order company; 191652 persons (rows) x 85 features (columns).
# - `Data_Dictionary.md`: Detailed information file about the features in the provided datasets.
# - `AZDIAS_Feature_Summary.csv`: Summary of feature attributes for demographics data; 85 features (rows) x 4 columns
# 
# Each row of the demographics files represents a single person, but also includes information outside of individuals, including information about their household, building, and neighborhood. You will use this information to cluster the general population into groups with similar demographic properties. Then, you will see how the people in the customers dataset fit into those created clusters. The hope here is that certain clusters are over-represented in the customers data, as compared to the general population; those over-represented clusters will be assumed to be part of the core userbase. This information can then be used for further applications, such as targeting for a marketing campaign.
# 
# To start off with, load in the demographics data for the general population into a pandas DataFrame, and do the same for the feature attributes summary. Note for all of the `.csv` data files in this project: they're semicolon (`;`) delimited, so you'll need an additional argument in your [`read_csv()`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_csv.html) call to read in the data properly. Also, considering the size of the main dataset, it may take some time for it to load completely.
# 
# Once the dataset is loaded, it's recommended that you take a little bit of time just browsing the general structure of the dataset and feature summary file. You'll be getting deep into the innards of the cleaning in the first major step of the project, so gaining some general familiarity can help you get your bearings.

# In[7]:


# Load in the general demographics data.
azdias_df = pd.read_csv("Udacity_AZDIAS_Subset.csv", delimiter=';')

# Load in the feature summary file.
feature_summary_df = pd.read_csv("AZDIAS_Feature_Summary.csv", delimiter=';')


# In[8]:


# Check the structure of the data after it's loaded (e.g. print the number of
# rows and columns, print the first few rows).

# Print the number of rows and columns
print("General Demographics Data Shape:", azdias_df.shape)



# In[9]:


# Print the first few rows
print("First few rows of General Demographics Data:")
print(azdias_df.head(5))


# In[10]:


pd.set_option("display.max_rows", None)
azdias_df.sample(15).T


# In[11]:


azdias_df.describe().T


# > **Tip**: Add additional cells to keep everything in reasonably-sized chunks! Keyboard shortcut `esc --> a` (press escape to enter command mode, then press the 'A' key) adds a new cell before the active cell, and `esc --> b` adds a new cell after the active cell. If you need to convert an active cell to a markdown cell, use `esc --> m` and to convert to a code cell, use `esc --> y`. 
# 
# ## Step 1: Preprocessing
# 
# ### Step 1.1: Assess Missing Data
# 
# The feature summary file contains a summary of properties for each demographics data column. You will use this file to help you make cleaning decisions during this stage of the project. First of all, you should assess the demographics data in terms of missing data. Pay attention to the following points as you perform your analysis, and take notes on what you observe. Make sure that you fill in the **Discussion** cell with your findings and decisions at the end of each step that has one!
# 
# #### Step 1.1.1: Convert Missing Value Codes to NaNs
# The fourth column of the feature attributes summary (loaded in above as `feat_info`) documents the codes from the data dictionary that indicate missing or unknown data. While the file encodes this as a list (e.g. `[-1,0]`), this will get read in as a string object. You'll need to do a little bit of parsing to make use of it to identify and clean the data. Convert data that matches a 'missing' or 'unknown' value code into a numpy NaN value. You might want to see how much data takes on a 'missing' or 'unknown' code, and how much data is naturally missing, as a point of interest.
# 
# **As one more reminder, you are encouraged to add additional cells to break up your analysis into manageable chunks.**

# In[12]:


# Checking for NaN 
azdias_df.isna().sum()


# In[13]:


# Identify missing or unknown data values and convert them to NaNs.
# Display the first couple of rows
print(feature_summary_df.head())


# In[14]:


# Check if DataFrame is loaded correctly
print('Feature Summary Shape', feature_summary_df.shape)


# In[15]:


# Looking at feature summary columns and check for missing entries
feature_summary_df.info()


# In[16]:


# Drop the column with lists before running describe()
feature_summary_df_filtered = feature_summary_df.drop(columns=['missing_values'], errors='ignore')

# Run describe safely
print(feature_summary_df_filtered.describe().T)


# In[17]:


# Extract the column that contains missing or unknown
missing_codes_column = feature_summary_df.iloc[:, 3]  # Adjust based on actual column name
print(missing_codes_column.head())  # View sample missing value codes


# In[18]:


# Define a function to parse missing value codes from string format
import ast  # Import Abstract Syntax Tree

def parse_missing_codes(value):
    try:
        return ast.literal_eval(value) if isinstance(value, str) else []
    except:
        return []

# Apply parsing to the fourth column
feature_summary_df['missing_values'] = feature_summary_df.iloc[:, 3].apply(parse_missing_codes)

# Display updated feature summary
print(feature_summary_df[['missing_values']].head())


# In[19]:


# Iterate through features and replace missing values with NaN

for index, row in feature_summary_df.iterrows():
    feature = row.iloc[0]  # Column name
    missing_values = row['missing_values']  # Extract parsed missing values

    # Debugging: Print feature name and missing values
    print(f"\nProcessing Feature: {feature}")
    print(f"Missing Values for {feature}: {missing_values}")

    # Ensure the feature exists in azdias_df and missing_values is a valid list
    if feature in azdias_df.columns and isinstance(missing_values, list) and len(missing_values) > 0:
        azdias_df.loc[:, feature] = azdias_df[feature].replace(missing_values, np.nan)
    else:
        print(f"Skipping {feature} (not found in azdias_df or invalid missing values list).")

# Check missing values after replacement
missing_summary = azdias_df.isnull().sum()
print("\nMissing Values per Column After Conversion:")
print(missing_summary[missing_summary > 0])


# #### Step 1.1.2: Assess Missing Data in Each Column
# 
# How much missing data is present in each column? There are a few columns that are outliers in terms of the proportion of values that are missing. You will want to use matplotlib's [`hist()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.hist.html) function to visualize the distribution of missing value counts to find these columns. Identify and document these columns. While some of these columns might have justifications for keeping or re-encoding the data, for this project you should just remove them from the dataframe. (Feel free to make remarks about these outlier columns in the discussion, however!)
# 
# For the remaining features, are there any patterns in which columns have, or share, missing data?

# In[20]:


# Perform an assessment of how much missing data there is in each column of the
# dataset.

# Step 1: Calculate missing values per column
missing_counts = azdias_df.isnull().sum()

# Step 2: Calculate missing value proportions (percentage)
missing_proportions = (missing_counts / len(azdias_df)) * 100

# Step 3: Store results in a DataFrame for better analysis
missing_stats = pd.DataFrame({
    "Missing Values": missing_counts,
    "Missing Percentage": missing_proportions
})

# Step 4: Sort by highest missing percentage
missing_stats_sorted = missing_stats.sort_values(by="Missing Percentage", ascending=False)

# Step 5: Plot histogram of missing values distribution
plt.figure(figsize=(12, 6))
plt.hist(missing_proportions, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel("Percentage of Missing Values per Column")
plt.ylabel("Number of Columns")
plt.title("Distribution of Missing Values Across Columns")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Step 6: Display missing data statistics
print("\nMissing Data Summary (Top 15 columns with highest missing values):")
print(missing_stats_sorted.head(15))


# In[21]:


###PROMPTED UDACITY AI for suggestions of how to investigate patterns
# Investigate patterns in the amount of missing data in each column.

# Plot heatmap to visualize missing values pattern
plt.figure(figsize=(12, 6))
sns.heatmap(azdias_df.isnull(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap in Azdias Dataset")
plt.show()


# In[22]:


azdias_df.info()


# In[23]:


# Remove the outlier columns from the dataset. (You'll perform other data
# engineering tasks such as re-encoding and imputation later.)

# Set threshold for dropping columns with excessive missing values (e.g., ≥50% missing)
threshold = 0.5 * len(azdias_df)

# Identify columns to drop
columns_to_drop = azdias_df.columns[azdias_df.isnull().sum() > threshold]

# Drop identified columns from the dataset
azdias_df_cleaned = azdias_df.drop(columns=columns_to_drop)

# Display removed columns and the new dataset shape
print(f"Removed {len(columns_to_drop)} columns due to high missing values.")
print(f"Remaining columns: {azdias_df_cleaned.shape[1]}")
print("Dropped Columns:", list(columns_to_drop))


# #### Discussion 1.1.2: Assess Missing Data in Each Column
# 
# (To improve the data quality and ensure our analysis is accurate, I removed columns with too many missing values (more than 50%). The columns I dropped were AGER_TYP, TITEL_KZ, KK_KUNDENTYP, and KBA05_BAUMAX because they had too much missing information to be helpful. TITEL_KZ, which represents personal titles like “Dr.,” had almost no data, making it unnecessary to keep. KK_KUNDENTYP, Customer Type Over the Past 12 Months, was missing more than 65% of its values, meaning it wouldn’t help much in identifying patterns. AGER_TYP, Best-ager typology, and KBA05_BAUMAX, which relates to  Common Building Type, both had over 50% missing values, making them unreliable for analysis. By removing these columns, I can focus on the data that is complete and meaningful, ensuring better results when identifying customer segments.)

# #### Step 1.1.3: Assess Missing Data in Each Row
# 
# Now, you'll perform a similar assessment for the rows of the dataset. How much data is missing in each row? As with the columns, you should see some groups of points that have a very different numbers of missing values. Divide the data into two subsets: one for data points that are above some threshold for missing values, and a second subset for points below that threshold.
# 
# In order to know what to do with the outlier rows, we should see if the distribution of data values on columns that are not missing data (or are missing very little data) are similar or different between the two groups. Select at least five of these columns and compare the distribution of values.
# - You can use seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) function to create a bar chart of code frequencies and matplotlib's [`subplot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.subplot.html) function to put bar charts for the two subplots side by side.
# - To reduce repeated code, you might want to write a function that can perform this comparison, taking as one of its arguments a column to be compared.
# 
# Depending on what you observe in your comparison, this will have implications on how you approach your conclusions later in the analysis. If the distributions of non-missing features look similar between the data with many missing values and the data with few or no missing values, then we could argue that simply dropping those points from the analysis won't present a major issue. On the other hand, if the data with many missing values looks very different from the data with few or no missing values, then we should make a note on those data as special. We'll revisit these data later on. **Either way, you should continue your analysis for now using just the subset of the data with few or no missing values.**

# In[24]:


# How much data is missing in each row of the dataset?

missing_per_row = azdias_df.isnull().sum(axis=1)


# In[25]:


# Display summary statistics
print(missing_per_row.describe())


# In[26]:


# Write code to divide the data into two subsets based on the number of missing
# values in each row.
threshold = 0.5 * azdias_df.shape[1]  # 50% of total columns


# In[27]:



low_missing_rows = azdias_df[azdias_df.isnull().sum(axis=1) <= threshold]  # Few missing values
high_missing_rows = azdias_df[azdias_df.isnull().sum(axis=1) > threshold]  # Many missing values

# Print dataset sizes
print(f"Low-Missing Rows: {low_missing_rows.shape[0]} (Kept)")
print(f"High-Missing Rows: {high_missing_rows.shape[0]} (Consider Removing)")


# In[28]:


# Compare the distribution of values for at least five columns where there are
# no or few missing values, between the two subsets.

# Select five columns with minimal missing values
selected_columns = low_missing_rows.dropna(axis=1).columns[:5]  # First 5 fully available columns

# Print selected columns
print("Selected Columns for Comparison:", selected_columns)


# In[29]:


def compare_distributions(df1, df2, columns):
    """
    Compares the distribution of values for selected columns
    between low-missing and high-missing row subsets.
    """
    fig, axes = plt.subplots(nrows=1, ncols=len(columns), figsize=(20, 5))

    for i, col in enumerate(columns):
        sns.countplot(x=col, data=df1, ax=axes[i], color="blue", alpha=0.6, label="Low Missing")
        sns.countplot(x=col, data=df2, ax=axes[i], color="red", alpha=0.6, label="High Missing")
        axes[i].set_title(f"Comparison of {col}")
        axes[i].legend()

    plt.tight_layout()
    plt.show()

# Compare distributions of selected columns
compare_distributions(low_missing_rows, high_missing_rows, selected_columns)


# #### Discussion 1.1.3: Assess Missing Data in Each Row
# 
# (Since the distributions of key features are similar between rows with high and low missing values, removing high-missing rows will not introduce significant bias. Dropping these rows simplifies data processing while preserving the overall structure and patterns in the dataset. If necessary, the high-missing rows can be analyzed separately later for further insights.)

# ### Step 1.2: Select and Re-Encode Features
# 
# Checking for missing data isn't the only way in which you can prepare a dataset for analysis. Since the unsupervised learning techniques to be used will only work on data that is encoded numerically, you need to make a few encoding changes or additional assumptions to be able to make progress. In addition, while almost all of the values in the dataset are encoded using numbers, not all of them represent numeric values. Check the third column of the feature summary (`feat_info`) for a summary of types of measurement.
# - For numeric and interval data, these features can be kept without changes.
# - Most of the variables in the dataset are ordinal in nature. While ordinal values may technically be non-linear in spacing, make the simplifying assumption that the ordinal variables can be treated as being interval in nature (that is, kept without any changes).
# - Special handling may be necessary for the remaining two variable types: categorical, and 'mixed'.
# 
# In the first two parts of this sub-step, you will perform an investigation of the categorical and mixed-type features and make a decision on each of them, whether you will keep, drop, or re-encode each. Then, in the last part, you will create a new data frame with only the selected and engineered columns.
# 
# Data wrangling is often the trickiest part of the data analysis process, and there's a lot of it to be done here. But stick with it: once you're done with this step, you'll be ready to get to the machine learning parts of the project!

# In[30]:


# How many features are there of each data type?
data_types = azdias_df.dtypes.value_counts()
print(data_types)


# In[31]:


# Identify categorical features (object type)
categorical_features = azdias_df.select_dtypes(include=['object']).columns

# Identify numerical features
numerical_features = azdias_df.select_dtypes(include=['int64', 'float64']).columns

# Display counts
print(f"Total Features: {azdias_df.shape[1]}")
print(f"Numeric/Ordinal Features: {len(numerical_features)}")
print(f"Categorical Features: {len(categorical_features)}")


# In[32]:


# Display unique values in object-type columns to identify mixed-type features
for col in categorical_features:
    print(f"{col}: {azdias_df[col].unique()[:10]}")  # Show first 10 unique values


# In[ ]:





# #### Step 1.2.1: Re-Encode Categorical Features
# 
# For categorical data, you would ordinarily need to encode the levels as dummy variables. Depending on the number of categories, perform one of the following:
# - For binary (two-level) categoricals that take numeric values, you can keep them without needing to do anything.
# - There is one binary variable that takes on non-numeric values. For this one, you need to re-encode the values as numbers or create a dummy variable.
# - For multi-level categoricals (three or more values), you can choose to encode the values using multiple dummy variables (e.g. via [OneHotEncoder](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html)), or (to keep things straightforward) just drop them from the analysis. As always, document your choices in the Discussion section.

# In[33]:


# Assess categorical variables: which are binary, which are multi-level, and
# which one needs to be re-encoded?
categorical_features = azdias_df.select_dtypes(include=['object']).columns

print("Categorical Features:", categorical_features)


# In[34]:


# Re-encode categorical variable(s) to be kept in the analysis.

# Convert 'OST_WEST_KZ' to numeric values
azdias_df['OST_WEST_KZ'] = azdias_df['OST_WEST_KZ'].map({'O': 0, 'W': 1})


# In[35]:


# Apply One-Hot Encoding
azdias_df = pd.get_dummies(azdias_df, columns=['CAMEO_DEUG_2015', 'CAMEO_DEU_2015'], drop_first=True)


# #### Discussion 1.2.1: Re-Encode Categorical Features
# 
# (During the categorical feature engineering process, I retained and transformed key categorical features to ensure they were numerically usable for clustering. The OST_WEST_KZ feature, which initially contained the categorical values "O" and "W", was converted into a binary numeric format where "O" was mapped to 0 and "W" to 1. Additionally, the multi-level categorical features CAMEO_DEUG_2015 & CAMEO_DEU_2015 were transformed using one-hot encoding, converting their multiple categorical values into separate binary columns while preserving their information. No categorical features were dropped, as encoding allowed retaining all valid data without loss. These preprocessing steps ensure the dataset is fully numerical, enabling effective clustering in the following steps. The next phase involves handling missing values and applying feature scaling for optimal model performance.)

# #### Step 1.2.2: Engineer Mixed-Type Features
# 
# There are a handful of features that are marked as "mixed" in the feature summary that require special treatment in order to be included in the analysis. There are two in particular that deserve attention; the handling of the rest are up to your own choices:
# - "PRAEGENDE_JUGENDJAHRE" combines information on three dimensions: generation by decade, movement (mainstream vs. avantgarde), and nation (east vs. west). While there aren't enough levels to disentangle east from west, you should create two new variables to capture the other two dimensions: an interval-type variable for decade, and a binary variable for movement.
# - "CAMEO_INTL_2015" combines information on two axes: wealth and life stage. Break up the two-digit codes by their 'tens'-place and 'ones'-place digits into two new ordinal variables (which, for the purposes of this project, is equivalent to just treating them as their raw numeric values).
# - If you decide to keep or engineer new features around the other mixed-type features, make sure you note your steps in the Discussion section.
# 
# Be sure to check `Data_Dictionary.md` for the details needed to finish these tasks.

# In[36]:


# Investigate "PRAEGENDE_JUGENDJAHRE" and engineer two new variables.

# Check unique values in PRAEGENDE_JUGENDJAHRE
print(azdias_df['PRAEGENDE_JUGENDJAHRE'].unique())


# In[37]:


# Handle missing values (optional: fill with mode)
azdias_df['PRAEGENDE_JUGENDJAHRE'].fillna(azdias_df['PRAEGENDE_JUGENDJAHRE'].mode()[0], inplace=True)

# Create 'DECADE' feature (extract first digit)
azdias_df['DECADE'] = (azdias_df['PRAEGENDE_JUGENDJAHRE'] // 10).astype(int)

# Create 'MOVEMENT_TYPE' feature (binary: 0 = Mainstream, 1 = Avantgarde)
azdias_df['MOVEMENT_TYPE'] = azdias_df['PRAEGENDE_JUGENDJAHRE'] % 2  # Odd = 1, Even = 0

# Drop the original mixed-type feature
azdias_df.drop(columns=['PRAEGENDE_JUGENDJAHRE'], inplace=True)


# In[38]:


# Check the new features
print(azdias_df[['DECADE', 'MOVEMENT_TYPE']].head())


# In[39]:


# Investigate "CAMEO_INTL_2015" and engineer two new variables.

 #Check unique values in CAMEO_INTL_2015
print(azdias_df['CAMEO_INTL_2015'].unique())


# In[40]:


# Replace 'XX' with NaN (for consistency in missing data)
azdias_df['CAMEO_INTL_2015'].replace(['XX','n', 'NaN'], np.nan, inplace=True)


# In[41]:


# Extract first digit as Wealth Level
azdias_df['WEALTH_LEVEL'] = azdias_df['CAMEO_INTL_2015'].str.extract('(\d)').astype(float)


# In[42]:


# Extract second digit as Life Stage
azdias_df['LIFE_STAGE'] = azdias_df['CAMEO_INTL_2015'].str.extract('.(\d)').astype(float)


# In[43]:


# Drop original column
azdias_df.drop(columns=['CAMEO_INTL_2015'], inplace=True)


# In[44]:


# Check new column
print(azdias_df[['WEALTH_LEVEL', 'LIFE_STAGE']].head())


# #### Discussion 1.2.2: Engineer Mixed-Type Features
# 
# (Multi-level categorical features CAMEO_DEUG_2015 and CAMEO_DEU_2015 were one-hot encoded to create multiple binary columns. The mixed-type feature PRAEGENDE_JUGENDJAHRE was split into two new variables: DECADE, representing the first digit as a numeric decade category, and MOVEMENT_TYPE, a binary indicator of mainstream (0) or avantgarde (1). Similarly, CAMEO_INTL_2015 was divided into WEALTH_LEVEL (first digit) and LIFE_STAGE (second digit), both treated as ordinal values. After extracting the necessary information, I dropped the original CAMEO_INTL_2015 and PRAEGENDE_JUGENDJAHRE columns.)

# #### Step 1.2.3: Complete Feature Selection
# 
# In order to finish this step up, you need to make sure that your data frame now only has the columns that you want to keep. To summarize, the dataframe should consist of the following:
# - All numeric, interval, and ordinal type columns from the original dataset.
# - Binary categorical features (all numerically-encoded).
# - Engineered features from other multi-level categorical features and mixed features.
# 
# Make sure that for any new columns that you have engineered, that you've excluded the original columns from the final dataset. Otherwise, their values will interfere with the analysis later on the project. For example, you should not keep "PRAEGENDE_JUGENDJAHRE", since its values won't be useful for the algorithm: only the values derived from it in the engineered features you created should be retained. As a reminder, your data should only be from **the subset with few or no missing values**.

# In[45]:


# If there are other re-engineering tasks you need to perform, make sure you
# take care of them here. (Dealing with missing data will come in step 2.1.)


# In[46]:


# Do whatever you need to in order to ensure that the dataframe only contains
# the columns that should be passed to the algorithm functions.

expected_features = ['DECADE', 'MOVEMENT_TYPE', 'WEALTH_LEVEL', 'LIFE_STAGE']
print(azdias_df[expected_features].head())


# ### Step 1.3: Create a Cleaning Function
# 
# Even though you've finished cleaning up the general population demographics data, it's important to look ahead to the future and realize that you'll need to perform the same cleaning steps on the customer demographics data. In this substep, complete the function below to execute the main feature selection, encoding, and re-engineering steps you performed above. Then, when it comes to looking at the customer data in Step 3, you can just run this function on that DataFrame to get the trimmed dataset in a single step.

# In[47]:


def clean_data(df):
    """
    Perform feature trimming, re-encoding, and engineering for demographics data.
    """

    # Convert 'OST_WEST_KZ' to numeric if present
    if 'OST_WEST_KZ' in df.columns:
        df['OST_WEST_KZ'] = df['OST_WEST_KZ'].map({'O': 0, 'W': 1})

    # One-Hot Encode Multi-Level Categorical Features ONLY IF THEY EXIST
    multi_level_features = ['CAMEO_DEUG_2015', 'CAMEO_DEU_2015']
    existing_features = [col for col in multi_level_features if col in df.columns]
    
    if existing_features:
        df = pd.get_dummies(df, columns=existing_features, drop_first=True)

    # Engineer Mixed-Type Features (Only if they exist)
    if 'PRAEGENDE_JUGENDJAHRE' in df.columns:
        df['DECADE'] = (df['PRAEGENDE_JUGENDJAHRE'] // 10).astype(float)
        df['MOVEMENT_TYPE'] = df['PRAEGENDE_JUGENDJAHRE'] % 2  # 0 = Mainstream, 1 = Avantgarde
    
    if 'CAMEO_INTL_2015' in df.columns:
        df['CAMEO_INTL_2015'].replace(['XX', 'n'], np.nan, inplace=True)
        df['WEALTH_LEVEL'] = df['CAMEO_INTL_2015'].astype(str).str[0].str.extract('(\d)').astype(float)
        df['LIFE_STAGE'] = df['CAMEO_INTL_2015'].astype(str).str[1].str.extract('(\d)').astype(float)

    # Drop original mixed-type features if they exist
    columns_to_drop = ['PRAEGENDE_JUGENDJAHRE', 'CAMEO_INTL_2015']
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')

    return df


# ## Step 2: Feature Transformation
# 
# ### Step 2.1: Apply Feature Scaling
# 
# Before we apply dimensionality reduction techniques to the data, we need to perform feature scaling so that the principal component vectors are not influenced by the natural differences in scale for features. Starting from this part of the project, you'll want to keep an eye on the [API reference page for sklearn](http://scikit-learn.org/stable/modules/classes.html) to help you navigate to all of the classes and functions that you'll need. In this substep, you'll need to check the following:
# 
# - sklearn requires that data not have missing values in order for its estimators to work properly. So, before applying the scaler to your data, make sure that you've cleaned the DataFrame of the remaining missing values. This can be as simple as just removing all data points with missing data, or applying an [Imputer](https://scikit-learn.org/0.16/modules/generated/sklearn.preprocessing.Imputer.html) to replace all missing values. You might also try a more complicated procedure where you temporarily remove missing values in order to compute the scaling parameters before re-introducing those missing values and applying imputation. Think about how much missing data you have and what possible effects each approach might have on your analysis, and justify your decision in the discussion section below.
# - For the actual scaling function, a [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html) instance is suggested, scaling each feature to mean 0 and standard deviation 1.
# - For these classes, you can make use of the `.fit_transform()` method to both fit a procedure to the data as well as apply the transformation to the data at the same time. Don't forget to keep the fit sklearn objects handy, since you'll be applying them to the customer demographics data towards the end of the project.

# In[ ]:


# If you've not yet cleaned the dataset of all NaN values, then investigate and
# do that now.
# Handling missing values 
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
azdias_cleaned = pd.DataFrame(imputer.fit_transform(azdias_df),
                              columns=azdias_df.columns)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
azdias_scaled = pd.DataFrame(scaler.fit_transform(azdias_cleaned),
                             columns=azdias_cleaned.columns)

#print("Number of features:", azdias_scaled.shape[1])

print("Number of features:", len(azdias_scaled.columns))
# Reduce the number of features to 110 Due to code disconnectint the kernel
# The idea do this was from https://knowledge.udacity.com/questions/1062294
#pca = PCA(n_components=100)
#azdias_pca = pca.fit_transform(azdias_scaled)

#azdias_pca_df = pd.DataFrame(
 #   azdias_pca, 
 #   columns=[f'PC{i+1}' for i in range(110)]
#)


# In[49]:


# Reduce the number of features to 110 Due to code disconnectint the kernel
# Idea from https://knowledge.udacity.com/questions/1062294


# ### Discussion 2.1: Apply Feature Scaling
# 
# (Applying Feature Scaling, we ensured that all the datasets' numbers were adjusted to be on the same scale. Some features (columns) might have had big numbers, while others had small ones, which could confuse the clustering algorithm. To fix this, we used a tool called StandardScaler, which changes the numbers so that they all have a mean (average) of 0 and a standard deviation of 1—basically, making sure no feature dominates just because of its size. .)

# ### Step 2.2: Perform Dimensionality Reduction
# 
# On your scaled data, you are now ready to apply dimensionality reduction techniques.
# 
# - Use sklearn's [PCA](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) class to apply principal component analysis on the data, thus finding the vectors of maximal variance in the data. To start, you should not set any parameters (so all components are computed) or set a number of components that is at least half the number of features (so there's enough features to see the general trend in variability).
# - Check out the ratio of variance explained by each principal component as well as the cumulative variance explained. Try plotting the cumulative or sequential values using matplotlib's [`plot()`](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html) function. Based on what you find, select a value for the number of transformed features you'll retain for the clustering part of the project.
# - Once you've made a choice for the number of components to keep, make sure you re-fit a PCA instance to perform the decided-on transformation.

# ##FIX ME##
# 
# # Convert transformed data to DataFrame for convenience
# ###  azdias_pca,
#     columns=[f'PC{i+1}' for i in range(110)]
# )
# 
# # Check the resulting shape and explained variance
# print(f"Reduced Data Shape: {azdias_pca_df.shape}")
# cumulative_variance = np.sum(pca.explained_variance_ratio_)
# print(f"Cumulative Explained Variance: {cumulative_variance:.2%}")
# 
# 

# In[50]:


# Investigate the variance accounted for by each principal component.


# In[51]:


# Re-apply PCA to the data while selecting for number of components to retain.


# ### Discussion 2.2: Perform Dimensionality Reduction
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding dimensionality reduction. How many principal components / transformed features are you retaining for the next step of the analysis?)

# ### Step 2.3: Interpret Principal Components
# 
# Now that we have our transformed principal components, it's a nice idea to check out the weight of each variable on the first few components to see if they can be interpreted in some fashion.
# 
# As a reminder, each principal component is a unit vector that points in the direction of highest variance (after accounting for the variance captured by earlier principal components). The further a weight is from zero, the more the principal component is in the direction of the corresponding feature. If two features have large weights of the same sign (both positive or both negative), then increases in one tend expect to be associated with increases in the other. To contrast, features with different signs can be expected to show a negative correlation: increases in one variable should result in a decrease in the other.
# 
# - To investigate the features, you should map each weight to their corresponding feature name, then sort the features according to weight. The most interesting features for each principal component, then, will be those at the beginning and end of the sorted list. Use the data dictionary document to help you understand these most prominent features, their relationships, and what a positive or negative value on the principal component might indicate.
# - You should investigate and interpret feature associations from the first three principal components in this substep. To help facilitate this, you should write a function that you can call at any time to print the sorted list of feature weights, for the *i*-th principal component. This might come in handy in the next step of the project, when you interpret the tendencies of the discovered clusters.

# In[52]:


# Map weights for the first principal component to corresponding feature names
# and then print the linked values, sorted by weight.
# HINT: Try defining a function here or in a new cell that you can reuse in the
# other cells.


# In[53]:


# Map weights for the second principal component to corresponding feature names
# and then print the linked values, sorted by weight.


# In[54]:


# Map weights for the third principal component to corresponding feature names
# and then print the linked values, sorted by weight.


# ### Discussion 2.3: Interpret Principal Components
# 
# (Double-click this cell and replace this text with your own text, reporting your observations from detailed investigation of the first few principal components generated. Can we interpret positive and negative values from them in a meaningful way?)

# ## Step 3: Clustering
# 
# ### Step 3.1: Apply Clustering to General Population
# 
# You've assessed and cleaned the demographics data, then scaled and transformed them. Now, it's time to see how the data clusters in the principal components space. In this substep, you will apply k-means clustering to the dataset and use the average within-cluster distances from each point to their assigned cluster's centroid to decide on a number of clusters to keep.
# 
# - Use sklearn's [KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans) class to perform k-means clustering on the PCA-transformed data.
# - Then, compute the average difference from each point to its assigned cluster's center. **Hint**: The KMeans object's `.score()` method might be useful here, but note that in sklearn, scores tend to be defined so that larger is better. Try applying it to a small, toy dataset, or use an internet search to help your understanding.
# - Perform the above two steps for a number of different cluster counts. You can then see how the average distance decreases with an increasing number of clusters. However, each additional cluster provides a smaller net benefit. Use this fact to select a final number of clusters in which to group the data. **Warning**: because of the large size of the dataset, it can take a long time for the algorithm to resolve. The more clusters to fit, the longer the algorithm will take. You should test for cluster counts through at least 10 clusters to get the full picture, but you shouldn't need to test for a number of clusters above about 30.
# - Once you've selected a final number of clusters to use, re-fit a KMeans instance to perform the clustering operation. Make sure that you also obtain the cluster assignments for the general demographics data, since you'll be using them in the final Step 3.3.

# In[55]:


# Over a number of different cluster counts...


    # run k-means clustering on the data and...
    
    
    # compute the average within-cluster distances.
    
    


# In[56]:


# Investigate the change in within-cluster distance across number of clusters.
# HINT: Use matplotlib's plot function to visualize this relationship.


# In[57]:


# Re-fit the k-means model with the selected number of clusters and obtain
# cluster predictions for the general population demographics data.


# ### Discussion 3.1: Apply Clustering to General Population
# 
# (Double-click this cell and replace this text with your own text, reporting your findings and decisions regarding clustering. Into how many clusters have you decided to segment the population?)

# ### Step 3.2: Apply All Steps to the Customer Data
# 
# Now that you have clusters and cluster centers for the general population, it's time to see how the customer data maps on to those clusters. Take care to not confuse this for re-fitting all of the models to the customer data. Instead, you're going to use the fits from the general population to clean, transform, and cluster the customer data. In the last step of the project, you will interpret how the general population fits apply to the customer data.
# 
# - Don't forget when loading in the customers data, that it is semicolon (`;`) delimited.
# - Apply the same feature wrangling, selection, and engineering steps to the customer demographics using the `clean_data()` function you created earlier. (You can assume that the customer demographics data has similar meaning behind missing data patterns as the general demographics data.)
# - Use the sklearn objects from the general demographics data, and apply their transformations to the customers data. That is, you should not be using a `.fit()` or `.fit_transform()` method to re-fit the old objects, nor should you be creating new sklearn objects! Carry the data through the feature scaling, PCA, and clustering steps, obtaining cluster assignments for all of the data in the customer demographics data.

# In[58]:


# Load in the customer demographics data.
customers = 


# In[ ]:


# Apply preprocessing, feature transformation, and clustering from the general
# demographics onto the customer data, obtaining cluster predictions for the
# customer demographics data.


# ### Step 3.3: Compare Customer Data to Demographics Data
# 
# At this point, you have clustered data based on demographics of the general population of Germany, and seen how the customer data for a mail-order sales company maps onto those demographic clusters. In this final substep, you will compare the two cluster distributions to see where the strongest customer base for the company is.
# 
# Consider the proportion of persons in each cluster for the general population, and the proportions for the customers. If we think the company's customer base to be universal, then the cluster assignment proportions should be fairly similar between the two. If there are only particular segments of the population that are interested in the company's products, then we should see a mismatch from one to the other. If there is a higher proportion of persons in a cluster for the customer data compared to the general population (e.g. 5% of persons are assigned to a cluster for the general population, but 15% of the customer data is closest to that cluster's centroid) then that suggests the people in that cluster to be a target audience for the company. On the other hand, the proportion of the data in a cluster being larger in the general population than the customer data (e.g. only 2% of customers closest to a population centroid that captures 6% of the data) suggests that group of persons to be outside of the target demographics.
# 
# Take a look at the following points in this step:
# 
# - Compute the proportion of data points in each cluster for the general population and the customer data. Visualizations will be useful here: both for the individual dataset proportions, but also to visualize the ratios in cluster representation between groups. Seaborn's [`countplot()`](https://seaborn.pydata.org/generated/seaborn.countplot.html) or [`barplot()`](https://seaborn.pydata.org/generated/seaborn.barplot.html) function could be handy.
#   - Recall the analysis you performed in step 1.1.3 of the project, where you separated out certain data points from the dataset if they had more than a specified threshold of missing values. If you found that this group was qualitatively different from the main bulk of the data, you should treat this as an additional data cluster in this analysis. Make sure that you account for the number of data points in this subset, for both the general population and customer datasets, when making your computations!
# - Which cluster or clusters are overrepresented in the customer dataset compared to the general population? Select at least one such cluster and infer what kind of people might be represented by that cluster. Use the principal component interpretations from step 2.3 or look at additional components to help you make this inference. Alternatively, you can use the `.inverse_transform()` method of the PCA and StandardScaler objects to transform centroids back to the original data space and interpret the retrieved values directly.
# - Perform a similar investigation for the underrepresented clusters. Which cluster or clusters are underrepresented in the customer dataset compared to the general population, and what kinds of people are typified by these clusters?

# In[ ]:


# Compare the proportion of data in each cluster for the customer data to the
# proportion of data in each cluster for the general population.


# In[ ]:


# What kinds of people are part of a cluster that is overrepresented in the
# customer data compared to the general population?


# In[ ]:


# What kinds of people are part of a cluster that is underrepresented in the
# customer data compared to the general population?


# ### Discussion 3.3: Compare Customer Data to Demographics Data
# 
# (Double-click this cell and replace this text with your own text, reporting findings and conclusions from the clustering analysis. Can we describe segments of the population that are relatively popular with the mail-order company, or relatively unpopular with the company?)

# > Congratulations on making it this far in the project! Before you finish, make sure to check through the entire notebook from top to bottom to make sure that your analysis follows a logical flow and all of your findings are documented in **Discussion** cells. Once you've checked over all of your work, you should export the notebook as an HTML document to submit for evaluation. You can do this from the menu, navigating to **File -> Download as -> HTML (.html)**. You will submit both that document and this notebook for your project submission.

# In[ ]:




