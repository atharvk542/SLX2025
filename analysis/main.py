import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from scipy.optimize import curve_fit
import textwrap

# --- Helper: Gini Coefficient ---
def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0:
        array -= np.amin(array)
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return (2 * np.sum(index * array)) / (n * np.sum(array)) - (n + 1) / n

# --- Load and Clean Data ---
file_path = "CombinedCO2Income.xlsx"
df = pd.read_excel(file_path, sheet_name="CO2Population")

# Clean column names
cols = [col.strip().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns]
df.columns = cols
print("Cleaned columns:", df.columns.tolist())

# Select key columns (Population column added for per capita analysis)
df = df[[ 
    'ZipCode', 
    'Population',
    'Total_Household_Carbon_Footprint_tCO2e/yr', 
    'HouseholdsPerZipCode', 
    'Total_Zip_Code_Carbon_Footprint_tCO2e/yr', 
    'Carbon_Footprint_per_capita_tCO2e/person', 
    'Households_Estimate_Mean_income_dollars',
    'Latitude',
    'Region'
]]

# Drop missing values
clean_df = df.dropna()

# Convert income and Population columns to numeric and drop conversion errors
clean_df['Households_Estimate_Mean_income_dollars'] = pd.to_numeric(
    clean_df['Households_Estimate_Mean_income_dollars'], errors='coerce'
)
clean_df['Population'] = pd.to_numeric(clean_df['Population'], errors='coerce')
clean_df = clean_df.dropna(subset=['Households_Estimate_Mean_income_dollars', 'Population'])

# Calculate emissions per household
clean_df['CO2_per_Household'] = clean_df['Total_Household_Carbon_Footprint_tCO2e/yr'] / clean_df['HouseholdsPerZipCode']

# --- Remove Extreme Outliers from CO2 data ---
lower_co2 = clean_df['CO2_per_Household'].quantile(0.05)
upper_co2 = clean_df['CO2_per_Household'].quantile(0.95)
clean_df = clean_df[(clean_df['CO2_per_Household'] >= lower_co2) & (clean_df['CO2_per_Household'] <= upper_co2)]

# --- Descriptive Stats per ZIP ---
descriptive_stats = clean_df.describe()
print("Descriptive Statistics:\n", descriptive_stats)

# --- Income Bracket Assignment ---
def income_bracket(income):
    if income < 50000:
        return 'Low'
    elif income < 150000:
        return 'Mid'
    else:
        return 'High'

clean_df['Income_Bracket'] = clean_df['Households_Estimate_Mean_income_dollars'].apply(income_bracket)

# --- Aggregated Views by Income Bracket ---
grouped = clean_df.groupby('Income_Bracket').agg({
    'CO2_per_Household': ['mean', 'median', 'count'],
    'Total_Household_Carbon_Footprint_tCO2e/yr': 'sum',
    'Households_Estimate_Mean_income_dollars': 'mean',
    'HouseholdsPerZipCode': 'sum'
})
print("\nAggregated Statistics by Income Bracket:\n", grouped)

total_households = clean_df['HouseholdsPerZipCode'].sum()
households_by_bracket = clean_df.groupby('Income_Bracket')['HouseholdsPerZipCode'].sum()
income_distribution_pct = (households_by_bracket / total_households * 100).round(2)
print("\nIncome Distribution (% of households):\n", income_distribution_pct)

# --- Correlation / Trend Metrics ---
r, p = pearsonr(clean_df['Households_Estimate_Mean_income_dollars'], clean_df['CO2_per_Household'])
print(f"\nPearson Correlation between Income and CO2 per Household: r = {r:.4f}, p-value = {p:.4g}")

# --- Inequality Measures ---
gini_income = gini(clean_df['Households_Estimate_Mean_income_dollars'])
gini_emissions = gini(clean_df['Total_Household_Carbon_Footprint_tCO2e/yr'])
print(f"\nGini Coefficient for Income: {gini_income:.4f}")
print(f"Gini Coefficient for Total Household CO2: {gini_emissions:.4f}")

# --- Plot: Income vs CO2 per Household (Scatter with Trend Line) ---
# Filter out data points with average income >= 1,000,000 dollars
filtered_df = clean_df[clean_df['Households_Estimate_Mean_income_dollars'] < 1_000_000]

plt.figure(figsize=(10,6))
sns.regplot(x='Households_Estimate_Mean_income_dollars', y='CO2_per_Household', 
            data=filtered_df, scatter_kws={'alpha':0.6}, line_kws={'color': 'red'})
plt.title("Income vs. CO2 Emissions per Household")
plt.xlabel("Average Household Income ($)")
plt.ylabel("CO2 Emissions per Household (tCO2e/year)")
plt.xlim(0, 400000)  # Zoom in on income range 0 to 400,000 dollars
plt.ylim(bottom=0)   # Ensure y-axis starts at 0
plt.grid(True)
plt.tight_layout()
plt.savefig("income_vs_co2_zoomed.png")

# --- Nonlinear Models ---
# Logarithmic Model
def log_model(x, a, b):
    return a * np.log(x) + b

# Exponential Model
def exp_model(x, a, b, c):
    return a * np.exp(b * x) + c

# Quadratic Model
def quad_model(x, a, b, c):
    return a * x**2 + b * x + c

# Filtered data for modeling (remove zero or negative income values for log/exp models)
model_df = filtered_df[filtered_df['Households_Estimate_Mean_income_dollars'] > 0]

# X and Y for modeling
X = model_df['Households_Estimate_Mean_income_dollars']
Y = model_df['CO2_per_Household']

# Fit Logarithmic Model
log_params, _ = curve_fit(log_model, X, Y)
log_fit = log_model(X, *log_params)

# Fit Exponential Model
exp_params, _ = curve_fit(exp_model, X, Y, maxfev=10000)
exp_fit = exp_model(X, *exp_params)

# Fit Quadratic Model
quad_params, _ = curve_fit(quad_model, X, Y)
quad_fit = quad_model(X, *quad_params)

# --- Plot: Income vs CO2 per Household with Nonlinear Models ---
plt.figure(figsize=(12, 8))
sns.scatterplot(x='Households_Estimate_Mean_income_dollars', y='CO2_per_Household', 
                data=model_df, alpha=0.6, label='Data')

# Linear Regression (already plotted earlier)
sns.regplot(x='Households_Estimate_Mean_income_dollars', y='CO2_per_Household', 
            data=model_df, scatter=False, line_kws={'color': 'red', 'label': 'Linear Fit'})

# Add Nonlinear Fits
plt.plot(X, log_fit, color='green', label='Logarithmic Fit')
plt.plot(X, exp_fit, color='orange', label='Exponential Fit')
plt.plot(X, quad_fit, color='blue', label='Quadratic Fit')

# Plot Formatting
plt.title("Income vs. CO2 Emissions per Household with Nonlinear Models")
plt.xlabel("Average Household Income ($)")
plt.ylabel("CO2 Emissions per Household (tCO2e/year)")
plt.xlim(0, 400000)  # Zoom in on income range 0 to 400,000 dollars
plt.ylim(bottom=0)   # Ensure y-axis starts at 0
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("income_vs_co2_nonlinear.png")

# --- Print Model Parameters ---
print("\nModel Parameters:")
print(f"Logarithmic Model: a = {log_params[0]:.4f}, b = {log_params[1]:.4f}")
print(f"Exponential Model: a = {exp_params[0]:.4f}, b = {exp_params[1]:.4f}, c = {exp_params[2]:.4f}")
print(f"Quadratic Model: a = {quad_params[0]:.4e}, b = {quad_params[1]:.4e}, c = {quad_params[2]:.4f}")

# --- Geographical Analysis ---

# Aggregate by Region to compare average CO2 emissions
regions_group = clean_df.groupby('Region').agg({
    'CO2_per_Household': ['mean', 'median', 'count'],
    'HouseholdsPerZipCode': 'sum'
})
print("\nAggregated Statistics by Region:\n", regions_group)

# Plot: Average CO2 per Household by Region
plt.figure(figsize=(8,6))
regions_avg = clean_df.groupby('Region')['CO2_per_Household'].mean().sort_values()
regions_avg.plot(kind='bar', color='skyblue')
plt.title("Average CO2 Emissions per Household by Region")
plt.xlabel("Region")
plt.ylabel("Average CO2 per Household (tCO2e/year)")
plt.tight_layout()
plt.savefig("co2_by_region.png")

# Correlation between Latitude and CO2 per Household
lat_co2_df = clean_df.dropna(subset=['Latitude'])
r_lat, p_lat = pearsonr(lat_co2_df['Latitude'], lat_co2_df['CO2_per_Household'])
print(f"\nPearson Correlation between Latitude and CO2 per Household: r = {r_lat:.4f}, p-value = {p_lat:.4g}")

# Scatter Plot: Latitude vs CO2 per Household
plt.figure(figsize=(10,6))
sns.scatterplot(x='Latitude', y='CO2_per_Household', data=lat_co2_df, alpha=0.6)
sns.regplot(x='Latitude', y='CO2_per_Household', data=lat_co2_df, scatter=False, line_kws={'color': 'red'})
plt.title("Latitude vs CO2 Emissions per Household")
plt.xlabel("Latitude")
plt.ylabel("CO2 per Household (tCO2e/year)")
plt.ylim(bottom=0)
plt.tight_layout()
plt.savefig("latitude_vs_co2.png")

# --- Population Analysis ---
# Using the Population column already in clean_df, compute per capita metrics

# Calculate CO2 per capita using total household CO2 divided by population
clean_df['CO2_per_capita'] = clean_df['Total_Household_Carbon_Footprint_tCO2e/yr'] / clean_df['Population']

# Create Population Bins (using quartiles)
clean_df['Population_Bin'] = pd.qcut(clean_df['Population'], q=4, labels=["Low Population", "Mid Population", "High Population", "Very High Population"])

# Aggregated metrics by Population Bin
pop_group = clean_df.groupby('Population_Bin').agg({
    'CO2_per_capita': ['mean', 'median', 'count'],
    'Population': 'sum'
})
pop_group.columns = ['_'.join(col).strip() for col in pop_group.columns.values]
print("\nAggregated Statistics by Population Bin:\n", pop_group)

# Export per capita summary to CSV
pop_group.to_csv("summary_statistics_by_population.csv")

# Plot: Average CO2 per Capita by Population Bin
plt.figure(figsize=(8,6))
pop_avg = clean_df.groupby('Population_Bin')['CO2_per_capita'].mean().sort_values()
pop_avg.plot(kind='bar', color='lightgreen')
plt.title("Average CO2 per Capita by Population Bin")
plt.xlabel("Population Bin")
plt.ylabel("Average CO2 per Capita (tCO2e per person)")
plt.tight_layout()
plt.savefig("co2_per_capita_by_population.png")

# Scatter Plot: Population vs CO2 per Capita
plt.figure(figsize=(10,6))
sns.scatterplot(x='Population', y='CO2_per_capita', data=clean_df, alpha=0.6)
sns.regplot(x='Population', y='CO2_per_capita', data=clean_df, scatter=False, line_kws={'color': 'red'})
plt.title("Population vs CO2 per Capita")
plt.xlabel("Population")
plt.ylabel("CO2 per Capita (tCO2e per person)")
plt.grid(True)
plt.tight_layout()
plt.savefig("population_vs_co2_per_capita.png")

# --- Additional Metrics by ZIP Code ---
# 1. Mean CO₂ / Household Distribution
plt.figure(figsize=(8,6))
sns.histplot(clean_df['CO2_per_Household'], bins=30, kde=True, color='violet')
plt.axvline(clean_df['CO2_per_Household'].mean(), color='red', linestyle='--', label='Mean')
plt.axvline(clean_df['CO2_per_Household'].median(), color='blue', linestyle='--', label='Median')
plt.title("Distribution of Mean CO₂ per Household")
plt.xlabel("CO₂ per Household (tCO2e/year)")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig("mean_co2_household_distribution.png")

# 2. Household Income Distribution (using mean income as proxy for median)
plt.figure(figsize=(8,6))
sns.histplot(clean_df['Households_Estimate_Mean_income_dollars'], bins=30, kde=True, color='salmon')
plt.title("Distribution of Household Income")
plt.xlabel("Household Income ($)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("household_income_distribution.png")

# 3. Percentage of High/Low Income Households (Income Skew)
income_bracket_pct = clean_df['Income_Bracket'].value_counts(normalize=True) * 100
plt.figure(figsize=(6,6))
income_bracket_pct.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'gold', 'lightgreen'])
plt.title("Income Bracket Distribution (% of ZIPs)")
plt.ylabel("")
plt.tight_layout()
plt.savefig("income_bracket_distribution_pie.png")

# 4. CO₂ per Capita Distribution
plt.figure(figsize=(8,6))
sns.histplot(clean_df['CO2_per_capita'], bins=30, kde=True, color='teal')
plt.title("Distribution of CO₂ per Capita")
plt.xlabel("CO₂ per Capita (tCO2e per person)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("co2_per_capita_distribution.png")

# 5. CO₂ per $1k Income Ratio
clean_df['CO2_per_1k_income'] = clean_df['CO2_per_Household'] / (clean_df['Households_Estimate_Mean_income_dollars'] / 1000)
plt.figure(figsize=(8,6))
sns.histplot(clean_df['CO2_per_1k_income'], bins=30, kde=True, color='darkorange')
plt.title("Distribution of CO₂ per $1k Income")
plt.xlabel("CO₂ per $1k Income (tCO2e per $1k)")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("co2_per_1k_income.png")

# 6. Top 10 ZIPs by Total Household CO₂ Emissions
top10_zip = clean_df.nlargest(10, 'Total_Household_Carbon_Footprint_tCO2e/yr')
plt.figure(figsize=(10,6))
sns.barplot(x='ZipCode', y='Total_Household_Carbon_Footprint_tCO2e/yr', data=top10_zip, palette='mako')
plt.title("Top 10 ZIP Codes by Total CO₂ Emissions")
plt.xlabel("ZIP Code")
plt.ylabel("Total Household CO₂ Emissions (tCO2e/year)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("top10_zip_emissions.png")

# 7. CO₂ Share by Income Tier
income_share = clean_df.groupby('Income_Bracket')['Total_Household_Carbon_Footprint_tCO2e/yr'].sum()
plt.figure(figsize=(6,6))
income_share.plot.pie(autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'gold', 'lightgreen'])
plt.title("CO₂ Share by Income Tier")
plt.ylabel("")
plt.tight_layout()
plt.savefig("co2_share_by_income_tier.png")

# --- CSV to Table Conversions ---

def csv_to_table_png(csv_file, output_png, max_col_width=20, wrap_len=15):
    """Load a CSV and render it as a wrapped text table PNG image."""
    df_csv = pd.read_csv(csv_file)

    # Wrap text in all cells including headers
    def wrap_cell(text):
        if isinstance(text, str):
            return '\n'.join(textwrap.wrap(text, width=wrap_len))
        else:
            return text

    df_wrapped = df_csv.applymap(wrap_cell)
    wrapped_columns = [wrap_cell(col) for col in df_csv.columns]

    # Set column widths proportionally
    col_width = max_col_width / 100
    col_widths = [col_width] * df_wrapped.shape[1]

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(df_wrapped.shape[1] * 2.2, df_wrapped.shape[0] * 0.6))
    ax.axis('tight')
    ax.axis('off')

    # Create the table
    table = ax.table(cellText=df_wrapped.values, colLabels=wrapped_columns,
                     loc='center', colWidths=col_widths)

    # Styling
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    # Optionally adjust row height
    for key, cell in table.get_celld().items():
        cell.set_height(0.25)

    fig.tight_layout()
    plt.savefig(output_png)
    plt.close()

# Convert the summary_statistics_by_population.csv to a table PNG
csv_to_table_png("summary_statistics_by_population.csv", "summary_statistics_by_population_table.png", wrap_len=12)

# If available, convert the summary_statistics_by_income.csv to a table PNG
try:
    csv_to_table_png("summary_statistics_by_income.csv", "summary_statistics_by_income_table.png", wrap_len=12)
except Exception as e:
    print("summary_statistics_by_income.csv not found or an error occurred:", e)

# --- Statistical Tests for Remaining Analyses ---

from scipy.stats import pearsonr, ttest_ind

# --- 1. Population Density vs. CO2 per Capita ---
# If land area is available: clean_df['Population_Density'] = clean_df['Population'] / clean_df['LandArea_sq_km']
# Else use proxy: population per household
clean_df['Population_Density'] = clean_df['Population'] / clean_df['HouseholdsPerZipCode']

# Compute Pearson correlation
r_pd, p_pd = pearsonr(clean_df['Population_Density'], clean_df['CO2_per_capita'])
print(f"\nPearson Correlation between Population Density and CO₂ per Capita: "
      f"r = {r_pd:.4f}, p-value = {p_pd:.4g}")
sig_pd = 'Highly Significant (p < 0.001)' if p_pd < 0.001 else (
         'Significant (p < 0.05)' if p_pd < 0.05 else 'Not Significant')
print("Significance:", sig_pd)

# --- 2. Income Bracket (Low vs High) vs. CO₂ per Household ---
low_co2  = clean_df.loc[clean_df['Income_Bracket']=='Low',  'CO2_per_Household']
high_co2 = clean_df.loc[clean_df['Income_Bracket']=='High', 'CO2_per_Household']

t_stat, p_ib = ttest_ind(low_co2, high_co2, equal_var=False)
mean_diff = low_co2.mean() - high_co2.mean()
print(f"\nMean CO₂ per Household — Low bracket: {low_co2.mean():.3f}, "
      f"High bracket: {high_co2.mean():.3f}, Difference (Low–High): {mean_diff:.3f}")
print(f"T-statistic = {t_stat:.3f}, p-value = {p_ib:.4g}")
sig_ib = 'Highly Significant (p < 0.001)' if p_ib < 0.001 else (
         'Significant (p < 0.05)' if p_ib < 0.05 else 'Not Significant')
print("Significance:", sig_ib)