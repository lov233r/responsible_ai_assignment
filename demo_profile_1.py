import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 1. Load the datasets
# --------------------------
# Read each CSV file and add a column indicating the county.
df_claiborne = pd.read_csv('Claiborne_county_synthetic_data.csv')
df_claiborne['County'] = 'Claiborne'

df_copiah = pd.read_csv('Copiah_county_synthetic_data.csv')
df_copiah['County'] = 'Copiah'

df_warren = pd.read_csv('Warren_county_synthetic_data.csv')
df_warren['County'] = 'Warren'

# Combine the three datasets into one DataFrame.
df = pd.concat([df_claiborne, df_copiah, df_warren], ignore_index=True)

# --------------------------
# 2. Aggregate the Data
# --------------------------
# Depending on your dataset, if each row is an individual, you can aggregate by counting the rows per category.
# If your dataset already has counts, adjust the aggregation accordingly.

# For racial distribution: count the number of individuals per county and race.
race_counts = df.groupby(['County', 'Race']).size().reset_index(name='Count')

# For gender distribution: count the number of individuals per county and gender.
gender_counts = df.groupby(['County', 'Gender']).size().reset_index(name='Count')

# --------------------------
# 3. Visualization Approaches
# --------------------------

# Define a bar width for grouped bar charts.
bar_width = 0.2

# ---- A. Grouped Bar Charts ----

# --- Racial Distribution Grouped Bar Chart ---
counties = race_counts['County'].unique()
races = race_counts['Race'].unique()
indices = np.arange(len(counties))

fig, ax = plt.subplots(figsize=(8, 6))
for i, race in enumerate(races):
    # For each county, extract the count for this race.
    counts = []
    for county in counties:
        count_val = race_counts[(race_counts['County'] == county) & (race_counts['Race'] == race)]['Count']
        counts.append(count_val.values[0] if not count_val.empty else 0)
    ax.bar(indices + i * bar_width, counts, bar_width, label=race)

ax.set_xlabel('County')
ax.set_ylabel('Count')
ax.set_title('Racial Distribution by County (Grouped Bar Chart)')
ax.set_xticks(indices + bar_width * (len(races) - 1) / 2)
ax.set_xticklabels(counties)
ax.legend()
plt.tight_layout()
plt.show()

# --- Gender Distribution Grouped Bar Chart ---
counties = gender_counts['County'].unique()
genders = gender_counts['Gender'].unique()
indices = np.arange(len(counties))

fig, ax = plt.subplots(figsize=(8, 6))
for i, gender in enumerate(genders):
    counts = []
    for county in counties:
        count_val = gender_counts[(gender_counts['County'] == county) & (gender_counts['Gender'] == gender)]['Count']
        counts.append(count_val.values[0] if not count_val.empty else 0)
    ax.bar(indices + i * bar_width, counts, bar_width, label=gender)

ax.set_xlabel('County')
ax.set_ylabel('Count')
ax.set_title('Gender Distribution by County (Grouped Bar Chart)')
ax.set_xticks(indices + bar_width * (len(genders) - 1) / 2)
ax.set_xticklabels(counties)
ax.legend()
plt.tight_layout()
plt.show()

# ---- B. Stacked Bar Charts ----

# --- Racial Distribution Stacked Bar Chart ---
# Pivot the race_counts DataFrame to have counties as index and races as columns.
df_race_pivot = race_counts.pivot(index='County', columns='Race', values='Count').fillna(0)

fig, ax = plt.subplots(figsize=(8, 6))
bottom = np.zeros(len(df_race_pivot.index))
for race in races:
    counts = df_race_pivot[race].values
    ax.bar(df_race_pivot.index, counts, bottom=bottom, label=race)
    bottom += counts  # update the bottom for stacking

ax.set_xlabel('County')
ax.set_ylabel('Count')
ax.set_title('Racial Distribution by County (Stacked Bar Chart)')
ax.legend()
plt.tight_layout()
plt.show()

# --- Gender Distribution Stacked Bar Chart ---
# Pivot the gender_counts DataFrame similarly.
df_gender_pivot = gender_counts.pivot(index='County', columns='Gender', values='Count').fillna(0)

fig, ax = plt.subplots(figsize=(8, 6))
bottom = np.zeros(len(df_gender_pivot.index))
for gender in genders:
    counts = df_gender_pivot[gender].values
    ax.bar(df_gender_pivot.index, counts, bottom=bottom, label=gender)
    bottom += counts

ax.set_xlabel('County')
ax.set_ylabel('Count')
ax.set_title('Gender Distribution by County (Stacked Bar Chart)')
ax.legend()
plt.tight_layout()
plt.show()
