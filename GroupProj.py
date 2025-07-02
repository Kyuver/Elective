##import necessary packages and libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import seaborn as sns

# Set global plot style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("="*80)
print("AI/ML GLOBAL SALARY TRENDS ANALYSIS")
print("="*80)

# ================================
# Load Dataset
# ================================
df = pd.read_csv('salaries.csv')

#Display overview of dataset contents 
print(f"\nDataset Overview:")
print(f"Total records: {len(df)}")
print(f"Date range: {df['work_year'].min()} - {df['work_year'].max()}")
print(f"Unique job titles: {df['job_title'].nunique()}")
print(f"Countries represented: {df['company_location'].nunique()}")

# ================================
# Data Cleaning
# ================================
#Display data table 
df.info()

#Check for missing values
print("\n========== Missing Values:==========")
print(df.isnull().sum())

#Remove exact duplicate rows 
df = df.drop_duplicates()

# Validate data types / Ensure salary column is numeric (in case of dirty entries)
df['salary_in_usd'] = pd.to_numeric(df['salary_in_usd'], errors='coerce')

# standardize Strings for job title format 
df['job_title'] = df['job_title'].str.lower().str.strip()


# ================================
# Outlier Removal (Salary using IQR)
# ================================
Q1 = df['salary_in_usd'].quantile(0.25)
Q3 = df['salary_in_usd'].quantile(0.75)
IQR = Q3 - Q1

# Define salary range using IQR
upper_bound = Q3 + 1.5 * IQR

# Remove outliers above upper bound (lower bound is not needed since salaries are positive)
initial_len = len(df)
df = df[df['salary_in_usd'] <= upper_bound]
removed = initial_len - len(df)

print(f"\n========== Outlier Removal ==========")
print(f"Removed {removed} outlier records with salary > ${upper_bound:,.0f}")
print(f"Remaining records: {len(df)}")


# ================================
# OBJECTIVE 1: Most In-Demand Jobs & Highest Salaries
# ================================
print("\n" + "="*80)
print("OBJECTIVE 1: MOST IN-DEMAND JOB TITLES & HIGHEST AVERAGE SALARIES")
print("="*80)

# Count number of job postings (demand)
job_counts = df['job_title'].value_counts()

# Compute average salary per job title
avg_salaries = df.groupby('job_title')['salary_in_usd'].mean()

# Combine both counts and averages for comprehensive analysis
job_summary = pd.DataFrame({
    'demand_count': job_counts,
    'average_salary_usd': avg_salaries,
    'total_salary_volume': job_counts * avg_salaries
}).fillna(0)

# Sort by demand (count) for most in-demand jobs
job_summary_by_demand = job_summary.sort_values(by='demand_count', ascending=False)

# Sort by salary for highest paying salaries
job_summary_by_salary = job_summary.sort_values(by='average_salary_usd', ascending=False)

#Display top 10 most in-demand job titles
print("\nTOP 10 MOST IN-DEMAND AI/ML JOB TITLES:")
print("-" * 60)
top_demand = job_summary_by_demand.head(10)
for i, (job, row) in enumerate(top_demand.iterrows(), 1):
    print(f"{i:2d}. {job:<35} | Demand: {row['demand_count']:3.0f} | Avg Salary: ${row['average_salary_usd']:8,.0f}")

#Display top 10 highest paying job titles
print("\nTOP 10 HIGHEST PAYING AI/ML JOB TITLES:")
print("-" * 60)
top_salary = job_summary_by_salary.head(10)
for i, (job, row) in enumerate(top_salary.iterrows(), 1):
    print(f"{i:2d}. {job:<35} | Avg Salary: ${row['average_salary_usd']:8,.0f} | Demand: {row['demand_count']:3.0f}")

# VISUALIZATION FOR OBJECTIVE 1 using horizontal bar charts
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

# Figure 1. Top 10 Most In-Demand Jobs
bars1 = ax1.barh(range(len(top_demand)), top_demand['demand_count'], color='steelblue', alpha=0.8)
ax1.set_yticks(range(len(top_demand)))
ax1.set_yticklabels([job[:25] + '...' if len(job) > 25 else job for job in top_demand.index], fontsize=9)
ax1.set_xlabel('Number of Job Postings')
ax1.set_title('Top 10 Most In-Demand AI/ML Positions', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

    # Add value labels
for i, (bar, count) in enumerate(zip(bars1, top_demand['demand_count'])):
        ax1.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                f'{int(count)}', ha='left', va='center', fontsize=8)

# Figure 2. Top 10 Highest Paying Jobs
bars2 = ax2.barh(range(len(top_salary)), top_salary['average_salary_usd'], color='darkgreen', alpha=0.8)
ax2.set_yticks(range(len(top_salary)))
ax2.set_yticklabels([job[:25] + '...' if len(job) > 25 else job for job in top_salary.index], fontsize=9)
ax2.set_xlabel('Average Salary (USD)')
ax2.set_title('Top 10 Highest Paying AI/ML Positions', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')
ax2.invert_yaxis()

    # Add value labels
for i, (bar, salary) in enumerate(zip(bars2, top_salary['average_salary_usd'])):
        ax2.text(bar.get_width() + 5000, bar.get_y() + bar.get_height()/2,
                f'${salary:,.0f}', ha='left', va='center', fontsize=8)

plt.tight_layout()
plt.show()

# ================================
# OBJECTIVE 2: Salary Prediction Model
# ================================
print("\n" + "="*80)
print("OBJECTIVE 2: SALARY PREDICTION MODEL DEVELOPMENT")
print("="*80)

# Define Features and target
features = ['experience_level', 'job_title']
target = 'salary_in_usd'

# Drop rows with missing values in relevant columns
df_model = df.dropna(subset=features + [target])
X = df_model[features]
y = df_model[target]

# Show model training information
print(f"\nModel Training Data:")
print(f"Training samples: {len(df_model)}")
print(f"Features used: {features}")
print(f"Experience levels: {sorted(df_model['experience_level'].unique())}")

# Preprocessing: One-hot encode categorical features for linear regression
categorical_features = ['experience_level', 'job_title']
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

# Create pipeline: preprocessor + linear regression model
model = make_pipeline(preprocessor, LinearRegression())

# Train/test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate model performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Display Evaluation results 
print(f"\nMODEL PERFORMANCE RESULTS:")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R² Score: {r2:.3f}")
print(f"Mean Absolute Error: ${np.mean(np.abs(y_test - y_pred)):,.2f}")

# VISUALIZATION FOR OBJECTIVE 2
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Figure 3. Scatter plot: actual vs. predicted salaries
ax1.scatter(y_test, y_pred, alpha=0.6, color='blue', s=50)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax1.set_xlabel('Actual Salary (USD)')
ax1.set_ylabel('Predicted Salary (USD)')
ax1.set_title(f'Model Prediction Accuracy\nR² = {r2:.3f}, RMSE = ${rmse:,.0f}', fontweight='bold')
ax1.grid(True, alpha=0.3)

# Figure 4. Residuals plot: error distribution
residuals = y_test - y_pred
ax2.scatter(y_pred, residuals, alpha=0.6, color='green', s=50)
ax2.axhline(y=0, color='r', linestyle='--', alpha=0.8)
ax2.set_xlabel('Predicted Salary (USD)')
ax2.set_ylabel('Residuals (Actual - Predicted)')
ax2.set_title('Residuals Analysis', fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ================================
# OBJECTIVE 3: Highest Paying Countries by Role
# ================================
print("\n" + "="*80)
print("OBJECTIVE 3: COUNTRIES WITH HIGHEST PAY BY ROLE")
print("="*80)

# Group by country and role to get average salary
country_role_salary = df.groupby(['company_location', 'job_title'])['salary_in_usd'].mean().reset_index()

# For each country, find the role with the highest average salary
highest_paid_roles = country_role_salary.loc[
    country_role_salary.groupby('company_location')['salary_in_usd'].idxmax()
]

# Sort to get globally highest paying country-role pairs
highest_paid_roles_sorted = highest_paid_roles.sort_values(by='salary_in_usd', ascending=False)

# Display top 15 highest paying country-role combinations
print("\nTOP 15 COUNTRIES WITH HIGHEST PAYING AI/ML ROLES:")
print("-" * 80)
top_15_countries = highest_paid_roles_sorted.head(15)
for i, (_, row) in enumerate(top_15_countries.iterrows(), 1):
    print(f"{i:2d}. {row['company_location']:3s} | ${row['salary_in_usd']:8,.0f} | {row['job_title']}")

# Additional analysis: country-level stats
country_stats = df.groupby('company_location').agg({
    'salary_in_usd': ['mean', 'median', 'count', 'std'],
    'job_title': 'nunique'
}).round(0)

country_stats.columns = ['avg_salary', 'median_salary', 'job_count', 'salary_std', 'unique_roles']
country_stats = country_stats.reset_index()

# Filter to countries with ≥10 job postings
significant_countries = country_stats[country_stats['job_count'] >= 10].sort_values('avg_salary', ascending=False)

# Display country summary 
print(f"\nCOUNTRY MARKET ANALYSIS (Countries with ≥10 job postings):")
print("-" * 80)
print(f"{'Country':<15} {'Avg Salary':<12} {'Jobs':<6} {'Roles':<6} {'Std Dev':<10}")
print("-" * 80)
for _, row in significant_countries.head(10).iterrows():
    print(f"{row['company_location']:<15} ${row['avg_salary']:<11,.0f} {row['job_count']:<6.0f} {row['unique_roles']:<6.0f} ${row['salary_std']:<9,.0f}")

# VISUALIZATION FOR OBJECTIVE 3
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

# Figure 5. Bar Chart: Top 15 highest paying countries with their premium roles
top_15 = highest_paid_roles_sorted.head(15)
bars1 = ax1.barh(range(len(top_15)), top_15['salary_in_usd'], color='darkred', alpha=0.8)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([f"{row['company_location']} - {row['job_title'][:20]}..." 
                    if len(row['job_title']) > 20 else f"{row['company_location']} - {row['job_title']}" 
                    for _, row in top_15.iterrows()], fontsize=10)
ax1.set_xlabel('Average Salary (USD)')
ax1.set_title('Top 15 Countries: Highest Paying AI/ML Roles', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='x')
ax1.invert_yaxis()

    # Add salary labels
for bar, salary in zip(bars1, top_15['salary_in_usd']):
        ax1.text(bar.get_width() + 10000, bar.get_y() + bar.get_height()/2,
                f'${salary:,.0f}', ha='left', va='center', fontsize=9)

# Figure 6. Bar Chart: Top 10 Countries by Average Salary
top_10_countries = significant_countries.head(10)
bars2 = ax2.bar(range(len(top_10_countries)), top_10_countries['avg_salary'], color='teal', alpha=0.8)
ax2.set_xticks(range(len(top_10_countries)))
ax2.set_xticklabels(top_10_countries['company_location'], rotation=45, ha='right')
ax2.set_ylabel('Average Salary (USD)')
ax2.set_title('Top 10 Countries by Average AI/ML Salary', fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels
for bar, salary in zip(bars2, top_10_countries['avg_salary']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3000,
                f'${salary:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# ================================
# COMPREHENSIVE SUMMARY
# ================================
print("\n" + "="*80)
print("COMPREHENSIVE ANALYSIS SUMMARY")
print("="*80)

print(f"\n OBJECTIVE 1 - KEY FINDINGS:")
print(f"   • Most in-demand role: {job_summary_by_demand.index[0]} ({job_summary_by_demand.iloc[0]['demand_count']:.0f} postings)")
print(f"   • Highest paying role: {job_summary_by_salary.index[0]} (${job_summary_by_salary.iloc[0]['average_salary_usd']:,.0f})")

print(f"\n OBJECTIVE 2 - MODEL PERFORMANCE:")
print(f"   • Model accuracy (R²): {r2:.1%}")
print(f"   • Prediction error (RMSE): ${rmse:,.0f}")
print(f"   • Model can explain {r2:.1%} of salary variance")

print(f"\n OBJECTIVE 3 - GLOBAL INSIGHTS:")
print(f"   • Top paying country: {highest_paid_roles_sorted.iloc[0]['company_location']} (${highest_paid_roles_sorted.iloc[0]['salary_in_usd']:,.0f})")
print(f"   • Most active market: {significant_countries.iloc[0]['company_location']} ({significant_countries.iloc[0]['job_count']:.0f} jobs)")

print(f"\n STRATEGIC RECOMMENDATIONS:")
print(f"   • For job seekers: Target '{job_summary_by_salary.index[0]}' roles in {highest_paid_roles_sorted.iloc[0]['company_location']}")
print(f"   • For employers: Expect high competition for '{job_summary_by_demand.index[0]}' roles")


print("\n" + "="*80)
