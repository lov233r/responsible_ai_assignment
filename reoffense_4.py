import pandas as pd

# --- Step 1: Load and Combine Your Data ---
# Assuming you have already loaded your three county CSV files and added a 'County' column.
# For example:
df_claiborne = pd.read_csv('Claiborne_county_synthetic_data.csv')
df_claiborne['County'] = 'Claiborne'

df_copiah = pd.read_csv('Copiah_county_synthetic_data.csv')
df_copiah['County'] = 'Copiah'

df_warren = pd.read_csv('Warren_county_synthetic_data.csv')
df_warren['County'] = 'Warren'

# Combine into one DataFrame
df = pd.concat([df_claiborne, df_copiah, df_warren], ignore_index=True)

# --- Step 2: Create a Binary Prediction Variable ---
# Assume that a bail decision of "Detained" means the judge predicted high risk (i.e. a positive prediction)
# and "Released" means low risk (i.e. a negative prediction).
df['prediction'] = df['Judge Decision']


# --- Step 3: Define a Function to Compute FPR and FNR ---
def compute_fairness_metrics(group):
    # Count True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN)
    TP = ((group['prediction'] == 1) & (group['Re-offense'] == 1)).sum()
    FP = ((group['prediction'] == 1) & (group['Re-offense'] == 0)).sum()
    TN = ((group['prediction'] == 0) & (group['Re-offense'] == 0)).sum()
    FN = ((group['prediction'] == 0) & (group['Re-offense'] == 1)).sum()

    # Calculate FPR and FNR, guarding against division by zero
    FPR = FP / (FP + TN) if (FP + TN) > 0 else None
    FNR = FN / (FN + TP) if (FN + TP) > 0 else None

    # Re-offense rate: proportion of individuals who actually reoffended
    reoffense_rate = group['Re-offense'].mean()

    return pd.Series({
        'Re-offense Rate': reoffense_rate,
        'FPR': FPR,
        'FNR': FNR
    })


# --- Step 4: Group by Race and Compute Metrics ---
fairness_metrics = df.groupby('Race').apply(compute_fairness_metrics)
print(fairness_metrics)
