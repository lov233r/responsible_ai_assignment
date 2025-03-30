import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Step 1: Load the Data
# -------------------------------
# Read each CSV file and add a "County" column for identification.
df_claiborne = pd.read_csv('Claiborne_county_synthetic_data.csv')
df_claiborne['County'] = 'Claiborne'

df_copiah = pd.read_csv('Copiah_county_synthetic_data.csv')
df_copiah['County'] = 'Copiah'

df_warren = pd.read_csv('Warren_county_synthetic_data.csv')
df_warren['County'] = 'Warren'

# Combine all data into one DataFrame
df = pd.concat([df_claiborne, df_copiah, df_warren], ignore_index=True)

# -------------------------------
# Step 2: Verify Column Names
# -------------------------------
# Ensure that the columns for race and gender are correctly named.
print(df.columns)
# If your gender column is named differently (e.g., "Sex"), adjust accordingly.

# -------------------------------
# Step 3: Analyze the Joint Distribution of Race and Gender
# -------------------------------
# Create a crosstab showing counts of defendants by County and Race, split by Gender.
race_gender_ct = pd.crosstab(index=[df['County'], df['Race']], columns=df['Gender'])
print("Counts:\n", race_gender_ct)

# Calculate percentage breakdown within each county.
# The grouping is by county, and percentages are computed across each county's total.
race_gender_pct = race_gender_ct.groupby(level=0).apply(lambda x: 100 * x / x.sum())
print("\nPercentages:\n", race_gender_pct)

# -------------------------------
# Step 4: Visualize the Distribution
# -------------------------------
# Option A: Aggregate countplot showing Race distribution with Gender as hue.
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Race', hue='Gender')
plt.title("Overall Distribution of Race and Gender Across Counties")
plt.ylabel("Count")
plt.show()

# Option B: Use a FacetGrid (catplot) to view each county separately.
g = sns.catplot(
    data=df,
    x='Race',
    hue='Gender',
    col='County',
    kind='count',
    height=4,
    aspect=0.8
)
g.set_titles("County: {col_name}")
g.set_axis_labels("Race", "Count")
plt.show()
