
                       ## "🔍📊 Analyzing TLC New Driver License Application Trends and Status Patterns in NYC 🚀" ##
                #  🐍 ──────────────────────────────────────────── ⋆⋅☆⋅⋆ ─────────────────────────────────────────────── 🐍  #


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, ttest_1samp, ttest_ind, ttest_rel

# ╰┈➤ 1. ⌛ Load the data
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

df = pd.read_excel("TLC_New_Driver_Application_Status.xlsx")
print(df)

# ╰┈➤ 2. 🔎 Basic information
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
print("Dataset Info:\n", df.info)     
print("First 5 column from the top: \n", df.head())
print("First 5 column from the bottom: \n", df.tail())

print("Describe: \n",df.describe());
print("Shape: ", df.shape)
print("Columns: \n", df.columns.tolist())     
print("Data type of each Column: \n", df.dtypes)


# ╰┈➤ 3. 🗐 Check for NULL and Duplicate values
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
print("Count of null values: \n", df.isna().sum())        
print("Count of duplicate values: \n", df.duplicated().sum())     


# ╰┈➤ 4. 🧹 Cleaning column names
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
print("\nColumns: \n", df.columns.tolist())

# ╰┈➤ 5. 🗑 Data Cleaning (Remove Duplicates, Handle Missing Values:
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
    # " But this dataset does not contains any Duplicate value and null values "
    # " If the dataset contains any duplicate and null values we can Handel them through the following code: "

print("\n Duplicate rows before cleaning:", df.duplicated().sum())
df.drop_duplicates(inplace=True)
print("\n Duplicate rows After cleaning:", df.duplicated().sum())

print("\n Shape before cleaning null values:", df.shape)
df.dropna(inplace=True)
print("\n Shape after cleaning null values:", df.shape)




# ╰┈➤ 6. 📈💰📊 Statistical Analysis & Insights
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

columns = ['app_no', 'type', 'app_date', 'status', 'fru_interview_scheduled', 'drug_test','wav_course',
          'defensive_driving', 'driver_exam', 'medical_clearance_form', 'other_requirements', 'last_updated']

    # frequency of each status
print("\n Frequency of each status: \n",df['status'].value_counts())

    # percentage of applications have each required document status
applied_columns = ['fru_interview_scheduled', 'drug_test', 'wav_course', 'defensive_driving',
                  'driver_exam', 'medical_clearance_form', 'other_requirements']

print("--------------------------------------------")
for col in applied_columns:
    print(f"\n{col} value counts in terms of Percentage:\n")
    print(df[col].value_counts(normalize = True) * 100)
    print("--------------------------------------------")

    # Average time(in terms of Days) to process an application

df["app_date"] = pd.to_datetime(df["app_date"])     # Converting string to date type
df["last_updated"] = pd.to_datetime(df["last_updated"])

df['processing_days'] = (df['last_updated'] - df['app_date']).dt.days
print(df['processing_days'].describe())     # Summary Statistics

    # mode (most common value) of each requirement
for col in columns:
    mode = df[col].mode()[0]
    print(f"\nMode of {col}: {mode}")


    # Variance and Standard Deviation of Application Processing Time
print("Standard Deviation:", df['processing_days'].std())
print("Variance:", df['processing_days'].var())

    # Group by type and get mean processing time

print(df.groupby('type')['processing_days'].mean())



# ╰┈➤ 7. 🚀📈🔍 Exploratory Data Analysis (EDA)
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈

    # Overall distribution of application statuses

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='status', hue='status', palette='viridis', legend=False)
plt.title('Distribution of Application Statuses')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

    # License Types Analysis
plt.figure(figsize=(10, 6))
df['type'].value_counts().plot(kind='pie', autopct='%1.1f%%', startangle=90, cmap="cividis")
plt.title('License Types Distribution')
plt.ylabel('')
plt.show()

    # Reasons for Pending/Denied Applications
pending_denied_df = df[df['status'].isin(['Pending', 'Denied'])]
plt.figure(figsize=(10, 8))
sns.countplot(data=pending_denied_df, x='other_requirements', hue='other_requirements', palette='coolwarm', legend=False)
plt.title('Reasons for Pending/Denied Applications')
plt.xlabel('Reason')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

    # Seasonality or Time Trends in Applications
df['app_date'] = pd.to_datetime(df['app_date'])
applications_by_month = df.groupby(df['app_date'].dt.month)['app_date'].count()
applications_by_month.plot(kind='line', figsize=(10, 6), marker='o', color='b')
plt.title('Monthly Application Volume')
plt.xlabel('Month')
plt.ylabel('Count')
plt.grid(True)
plt.show()

    # Average Processing Time
plt.figure(figsize=(10, 6))
sns.histplot(df['processing_days'], bins=20, color='g', kde=True)
plt.title('Distribution of Processing Times')
plt.xlabel('Days')
plt.ylabel('Frequency')
plt.show()


    # Anomalies in Incomplete Applications
incomplete_df = df[df['status'] == 'Incomplete']
sns.heatmap(incomplete_df.isnull(), cbar=False, cmap="coolwarm")
plt.title('Heatmap of Missing Values in Incomplete Applications')
plt.show()



# ╰┈➤ 7. 🚀📈🔍 Hypothesis Testing:
#┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈
# Z-Test
# Assume population standard deviation and perform Z-test
def z_test():
    sample_mean = df['processing_days'].mean()
    population_std = 5  # Replace with known population standard deviation
    sample_size = len(df['processing_days'])
    
    z_score = (sample_mean - 10) / (population_std / np.sqrt(sample_size))  # Testing against the value of 10 days
    p_value = 2 * (1 - norm.cdf(abs(z_score)))  # Two-tailed test
    
    print("\n--- Z-Test Results ---")
    print(f"Z-Score: {z_score}")
    print(f"P-Value: {p_value}")
    if p_value < 0.05:
        print("Reject the Null Hypothesis")
    else:
        print("Fail to Reject the Null Hypothesis")

# 2. Hypothesis Testing: One-Sample t-Test
def one_sample_t_test():
    t_stat, p_value = ttest_1samp(df['processing_days'], 10)  # Testing against the value of 10 days
    
    print("\n--- One-Sample t-Test Results ---")
    print(f"T-Statistic: {t_stat}")
    print(f"P-Value: {p_value}")
    if p_value < 0.05:
        print("Reject the Null Hypothesis")
    else:
        print("Fail to Reject the Null Hypothesis")
        
# Hypothesis Testing: Independent Two-Sample t-Test
def two_sample_t_test():
    # Check sample sizes to prevent errors
    type_a = df[df['type'] == 'Type A']['processing_days']
    type_b = df[df['type'] == 'Type B']['processing_days']

    print("\n--- Two-Sample t-Test Results ---")
    print("Sample size for Type A:", len(type_a))
    print("Sample size for Type B:", len(type_b))
    
    if len(type_a) > 1 and len(type_b) > 1:  # Ensure there are enough samples
        t_stat, p_value = ttest_ind(type_a, type_b, equal_var=False)  # Assume unequal variance
        print("\n--- Two-Sample t-Test Results ---")
        print(f"T-Statistic: {t_stat}")
        print(f"P-Value: {p_value}")
        if p_value < 0.05:
            print("Reject the Null Hypothesis")
        else:
            print("Fail to Reject the Null Hypothesis")
    else:
        print("Insufficient data in one or both groups for Two-Sample t-Test.")


# Execute all tests
z_test()
one_sample_t_test()
two_sample_t_test()


    


