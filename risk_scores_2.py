import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Read each CSV file
df_claiborne = pd.read_csv('Claiborne_county_synthetic_data.csv')
df_copiah    = pd.read_csv('Copiah_county_synthetic_data.csv')
df_warren    = pd.read_csv('Warren_county_synthetic_data.csv')

# Tag each DataFrame with a 'County' column
df_claiborne['County'] = 'Claiborne'
df_copiah['County']    = 'Copiah'
df_warren['County']    = 'Warren'

# Combine into a single DataFrame
df = pd.concat([df_claiborne, df_copiah, df_warren], ignore_index=True)

sns.set_style('whitegrid')  # Just a nicer background style

# Create a catplot with columns = County, x-axis = Race, y-axis = Risk Score
# Hue = Gender to separate distributions by gender.
g = sns.catplot(
    data=df,
    x='Race',
    y='Risk Score',
    hue='Gender',
    col='County',
    kind='box',        # or 'violin' for a violin plot
    height=4,
    aspect=1
)

# Rotate x-axis labels if they are too long
g.set_xticklabels(rotation=30)

g.fig.subplots_adjust(bottom=0.3)

g.fig.legend(title='Gender',
             loc='lower center',
             bbox_to_anchor=(0.5, -0.05),
             ncol=len(df['Gender'].unique()))

plt.tight_layout()
plt.show()
