# Healthcare Claims & Patient Data Analysis for Fraud, Waste, and Abuse (FWA) Detection

## Introduction

This project aims to analyze healthcare claims data to identify potential Fraud, Waste, and Abuse (FWA) patterns. Using synthetic datasets generated via Synthea, we perform exploratory data analysis, feature engineering, anomaly detection, and visualization to uncover suspicious behaviors from patient, provider, and claims perspectives.

---

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Jupyter Notebook
- Required Python packages:

```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### Initializing the Notebook
Clone or download this repository.

Launch Jupyter Notebook in your project directory:
```bash
jupyter notebook
```
Open the notebook **healthcare_fwa_analysis.ipynb**.

### Dataset Description

The datasets used include:

- **Patients Data (patients.csv):** Demographics, medical history, and socioeconomic info.

- **Claims Data (claims.csv):** Claims details such as amount, date, diagnoses, and provider.

- **Claims Transactions (claims_transactions.csv):** Detailed transaction info linked to claims.

- **Providers Data (providers.csv):** Provider specialties, locations, and contact info.

The datasets are synthetically generated and stored in CSV format, providing a realistic basis for analysis.

## Data Loading and Preprocessing

```bash
import pandas as pd

patients_df = pd.read_csv('path/to/patients.csv')
claims_df = pd.read_csv('path/to/claims.csv')
transactions_df = pd.read_csv('path/to/claims_transactions.csv')
providers_df = pd.read_csv('path/to/providers.csv')

# Convert date columns to datetime
patients_df['BIRTHDATE'] = pd.to_datetime(patients_df['BIRTHDATE'], errors='coerce')
claims_df['SERVICEDATE'] = pd.to_datetime(claims_df['SERVICEDATE'], errors='coerce')
transactions_df['FROMDATE'] = pd.to_datetime(transactions_df['FROMDATE'], errors='coerce')
transactions_df['TODATE'] = pd.to_datetime(transactions_df['TODATE'], errors='coerce')
```

## Exploratory Data Analysis (EDA)

### 1. Patients Demographics
#### 1.1 Distribution of Gender
```bash
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=patients_df, x='GENDER')
plt.title('Patient Gender Distribution')
plt.show()
```

#### 1.2 Age Calculation
```bash
import datetime as dt
today = dt.date.today()
patients_df['AGE'] = patients_df['BIRTHDATE'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
sns.histplot(patients_df['AGE'], bins=20)
plt.title('Patient Age Distribution')
plt.show()
```
#### 1.3 Claims Overview

```bash
# Claim amount distribution
sns.boxplot(x=claims_df['AMOUNT'])
plt.title('Claim Amounts')
plt.show()

# Claims over Time
claims_df['SERVICEDATE'].hist(bins=30)
plt.title('Claims Over Time')
plt.show()
```

### 1.4 Provider Analysis

```bash
# Top providers by number of claims
provider_claim_counts = claims_df['PROVIDERID'].value_counts().head(10)
provider_claim_counts.plot(kind='bar')
plt.ylabel('Number of Claims')
plt.title('Top 10 Providers by Claim Count')
plt.show()
```

## Feature Engineering & Suspicious Pattern Detection


#### 1. Identify High-Value Claims

```bash
high_claim_threshold = claims_df['AMOUNT'].quantile(0.99)
suspicious_claims = claims_df[claims_df['AMOUNT'] > high_claim_threshold]
```

#### 2. Detect Repetitive & Rapid Claims
```bash
claims_df['DATE'] = claims_df['SERVICEDATE'].dt.date
claims_per_provider_day = claims_df.groupby(['PROVIDERID', 'DATE']).size()
suspicious_frequency = claims_per_provider_day[claims_per_provider_day > 10]
```
#### 3. Anomaly Detection with Isolation Forest

```bash
from sklearn.ensemble import IsolationForest
import numpy as np

# Prepare features
features = claims_df[['AMOUNT']]
features.fillna(0, inplace=True

clf = IsolationForest(contamination=0.01, random_state=42)
claims_df['anomaly_score'] = clf.fit_predict(features)

# Filter suspected fraudulent claims
suspects = claims_df[claims_df['anomaly_score'] == -1]
```
### Visualization of Suspicious Activities

```bash
import networkx as nx

G = nx.Graph()
for _, row in suspects.iterrows():
    G.add_node(row['PROVIDERID'], type='provider')
    G.add_node(row['PATIENTID'], type='patient')
    G.add_edge(row['PROVIDERID'], row['PATIENTID'])

nx.draw(G, with_labels=True)
plt.title('Suspected Provider-Patient Network')
plt.show()
```

## Concepts and Techniques Used

> **Outlier Detection Techniques:**  Isolation Forests identify anomalies based on isolation in feature space.

> **Feature Engineering:** Claim amounts, claim frequency, and temporal patterns.

> **Visualization:** Box plots, histograms, network graphs to visualize suspicious activity.

> **Thresholding:** Using statistical thresholds (e.g., 99th percentile) to flag high claims.

> **Network Analysis:** Exploring relationships between providers and patients.

## Findings & Next Steps
> High claim amounts and frequent claims flagged as potential fraud.

> Outlier claims identified via anomaly detection.

> Network analysis reveals clusters of providers and patients with suspicious interactions.

> Next steps include integrating supervised learning models with labeled data, refining feature sets, and developing dashboards for ongoing monitoring.

## Final Notes
> This analysis serves as an initial screening tool.

> False positives are possible; manual review is essential.

> Models should be continuously updated with new data and feedback.


### References & Resources
- Synthea: Synthetic Healthcare Data Generation
    - Main GitHub repo: https://github.com/synthetichealth/synthea
    - International extension: https://github.com/synthetichealth/synthea-international

- Generating Data for Ireland (IE):
    - Synthea can be customized for Ireland’s healthcare systems, demographics, and coding standards.

- See README.md on respective repo for for setup instructions.
- Documentation & Code: Synthea documentation: https://github.com/synthetichealth/synthea/wiki


### License
This project is licensed under the PRIVATE & COPYRIGHTED License. See the LICENSE file for details.

**© Neural Nexus Team** — All rights reserved.

With love for healthcare data analysis and fraud detection.
