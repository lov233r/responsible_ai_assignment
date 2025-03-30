import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Example of loading and combining data:
df_claiborne = pd.read_csv('Claiborne_county_synthetic_data.csv')
df_claiborne['County'] = 'Claiborne'

df_copiah = pd.read_csv('Copiah_county_synthetic_data.csv')
df_copiah['County'] = 'Copiah'

df_warren = pd.read_csv('Warren_county_synthetic_data.csv')
df_warren['County'] = 'Warren'

df = pd.concat([df_claiborne, df_copiah, df_warren], ignore_index=True)

# Check the columns and convert if necessary
df['Bail Decision'] = df['Judge Decision'].astype(str)
df['Race'] = df['Race'].astype(str)
df['Gender'] = df['Gender'].astype(str)



sns.set_style('whitegrid')

# Create a box plot: Each county's plot is faceted by Race,
# with bail decisions on the x-axis and risk score on the y-axis.
g = sns.catplot(
    data=df,
    x='Bail Decision',
    y='Risk Score',
    hue='Gender',
    col='Race',
    kind='box',       # change to 'violin' if preferred
    height=4,
    aspect=0.8
)

g.set_xticklabels(rotation=30)
g.fig.subplots_adjust(bottom=0.2)  # Ensure enough room for legends if needed

plt.show()

