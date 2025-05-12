# Healthcare Fraud Detection Using Graph Analysis and AI

> **Note:**
This repository contains our project for the **ITAG Atlantec Hackathon 2025** held in Galway. Our team (The Neural Nexus) developed an innovative approach to detect healthcare fraud, waste, and abuse by leveraging **graph analysis** and **machine learning** techniques. The goal is to improve detection accuracy, transparency, and investigation efficiency in healthcare systems.

## Introduction

This project aims to analyze healthcare claims data to identify potential Fraud, Waste, and Abuse (FWA) patterns. Using synthetic datasets generated via Synthea, we perform exploratory data analysis, feature engineering, anomaly detection, and visualization to uncover suspicious behaviors from patient, provider, and claims perspectives.

### About Fraud, Waste, and Abuse (FWA)
Fraud, Waste, and Abuse (FWA) in healthcare represent significant challenges worldwide, leading to billions of euros in unnecessary costs annually. Specifically, in Ireland and across the European Union, healthcare systems are under increasing pressure to optimize resources while maintaining high standards of care.

- **Fraud** involves intentional deception or misrepresentation for financial gain, such as falsifying claims or identities.
- **Waste** refers to overutilization or inefficient practices that increase costs without improving patient outcomes.
- **Abuse** includes practices like upcoding, billing for services not rendered, or unbundling procedures to inflate charges.

### Statistics and Impact:

- According to the European Healthcare Fraud and Corruption Network (EHFCN), EU member states lose an estimated **€56 billion annually** due to healthcare fraud and abuse. [Source](https://www.ehfcn.eu/)
- In Ireland, the Health Service Executive (HSE) estimates that **up to 10%** of healthcare expenditure may be attributable to fraud, waste, or abuse, translating into hundreds of millions of euros annually.
- The **World Bank** reports that health sector fraud costs can account for **up to 3-5% of total health expenditures** in developed countries.

### Why is detecting FWA important?
Effective detection helps prevent financial losses, ensures equitable resource distribution, and maintains the integrity of healthcare systems. This project leverages data analysis and machine learning techniques to identify suspicious claims and activities in synthetic Irish healthcare data.

---

## Analysis & Techniques for FWA Detection
### 1. Understanding FWA Concepts
| Concept | Explanation                              | Indicator                                 | Formula / Metric             | Interpretation                           |
|---------|------------------------------------------|-------------------------------------------|------------------------------|------------------------------------------|
| Fraud   | Intentional deception for financial gain | Claims with false info or staging         | False claims, identity theft | Excessive claims, suspicious patterns    |
| Waste   | Overutilization or misuse of resources   | Excessive billing, unnecessary procedures | Cost per patient/provider    | High expenses with no added value        |
| Abuse   | Improper billing practices               | Upcoding, unbundling services             | Billing codes misuse         | Discrepancies between services and codes |


### 2. Techniques & Code
- **Outlier Detection** (e.g., Isolation Forest)
    - **Purpose:** Automatically flag claims that deviate significantly from normal patterns.
    - **Formula / Concept:** Isolation Forest isolates anomalies by randomly partitioning the data space. Fewer splits to isolate a point suggest an anomaly.
    - **Interpretation:** Claims flagged as -1 are potential frauds or anomalies needing manual review.
- **Threshold-based Flagging** - High Claim Amounts:
    - **Purpose:** Indicates potentially fraudulent claims with abnormally high charges.
- **Visualization & Network Analysis**
    - **Purpose:** This reveals clusters of suspicious providers and patients.


## Getting Started

### Overview

The repository includes three main notebooks:

- `healthcare_fwa_analysis.ipynb`  
  Performs an initial analysis of healthcare claims data based on basic provider and patient network analysis.

- `healthcare_fwa_graph_analysis.ipynb`  
  Implements a graph-based approach as described in relevant research papers.

- `healthcare_fwa_opt2_analysis.ipynb`  
  Utilizes machine learning techniques to detect suspicious claims indicative of fraud or abuse.

---

Directory Structure
```.
├── notebooks/
│   ├── healthcare_fwa_analysis.ipynb
│   ├── healthcare_fwa_graph_analysis.ipynb
│   └── healthcare_fwa_opt2_analysis.ipynb
├── data/
│   └── sample_data/
│       └── csv/
│           └── [county folders]
├── containers/
│   ├── Dockerfile
│   └── other container configs
├── requirements.txt
└── README.md
```

### Datasets
This project utilizes a **synthetic healthcare dataset** generated through the **Synthea** tool. Synthea is an open-source simulator designed to produce realistic, anonymized patient records based on population models. This approach enables scalable and privacy-preserving research and analysis.

#### Why Use Synthetic Data?

- **Privacy & Confidentiality:** Synthetic data circumvents privacy concerns associated with real patient records.
- **Controlled Environment:** Facilitates testing, experimentation, and validation without risking sensitive information.
- **Customizability:** Data can be tailored to specific scenarios, demographic distributions, or rare conditions, enhancing research flexibility.
- **Resource Efficiency:** Eliminates the need for complex data access procedures and compliance hurdles.

The datasets used include:

- **Patients Data (patients.csv):** Demographics, medical history, and socioeconomic info.
- **Claims Data (claims.csv):** Claims details such as amount, date, diagnoses, and provider.
- **Claims Transactions (claims_transactions.csv):** Detailed transaction info linked to claims.
- **Providers Data (providers.csv):** Provider specialties, locations, and contact info.

The datasets are synthetically generated and stored in CSV format, providing a realistic basis for analysis.

The datasets are stored in the following directory structure:

```
data/
└── sample_data/
    └── csv/
        └── [county-specific folders]/
            └── *.csv
```
Each folder contains CSV files specific to a county. (for e.g: galway, limerick, dublin, cork) 

> **Warning:**
Using Synthea-generated synthetic datasets enables safe and flexible healthcare data analysis. However, it is crucial to understand and account for the inherent biases to ensure meaningful and responsible insights.


### Prerequisites

Ensure you have the following installed:
- Google Cloud Platform account ( for using Vertex AI workbench to use these notebooks)
- Python 3.8+ environment (if running notebooks directly)
- Basic familiarity with Jupyter Notebooks

#### Required Python packages:
Install the following packages within jupyter notebooks or python code.
```bash
pip install pandas numpy matplotlib seaborn plotly scikit-learn
```

### Clone the repo
```bash
git clone https://github.com/HackmaniaGX/neural-nexus-healthcare-fwa-analysis.git
cd neural-nexus-healthcare-fwa-analysis
```

### Set Up Environment
#### Using Docker
To build and run the container:
```bash
docker build -t healthcare-fwa .
docker run -p 8888:8888 -v "$(pwd):/app" healthcare-fwa
```

This will start Jupyter Notebook server accessible at http://localhost:8888.

#### Manual Setup
Alternatively, set up a Python environment:

```bash
pip install -r requirements.txt
```

### Running the Notebooks

Launch Jupyter Notebook in project directory:
```bash
jupyter notebook
```
- Open the notebooks in the notebooks/ directory:

- Initial Analysis: Use **healthcare_fwa_analysis.ipynb**

- Graph-Based ML Analysis: **healthcare_fwa_graph_analysis.ipynb**

- ML-based Detection Algorithm: Use **healthcare_fwa_opt2_analysis.ipynb*

> **Note**
Update dataset paths if necessary to point to your local data folders.

## Analysis Process 

Here we are trying to document the few high level steps involved in processing the data. 

### Data Loading and Preprocessing

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

### Data Merging

```bash
    # Merge
    merged_df = claims_df.merge(transactions_df, on='CLAIMID', how='left')
```


### Exploratory Data Analysis (EDA)

#### 1. Patients Demographics
##### 1.1 Distribution of Gender
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(data=patients_df, x='GENDER')
plt.title('Patient Gender Distribution')
plt.show()
```

##### 1.2 Age Calculation
```python
import datetime as dt
today = dt.date.today()
patients_df['AGE'] = patients_df['BIRTHDATE'].apply(lambda x: today.year - x.year - ((today.month, today.day) < (x.month, x.day)))
sns.histplot(patients_df['AGE'], bins=20)
plt.title('Patient Age Distribution')
plt.show()
```
##### 1.3 Claims Overview

```python
# Claim amount distribution
sns.boxplot(x=merged_df['AMOUNT'])
plt.title('Claim Amounts')
plt.show()

# Claims over Time
merged_df['SERVICEDATE'].hist(bins=30)
plt.title('Claims Over Time')
plt.show()
```

#### 1.4 Provider Analysis

```python
# Top providers by number of claims
provider_claim_counts = merged_df['PROVIDERID'].value_counts().head(10)
provider_claim_counts.plot(kind='bar')
plt.ylabel('Number of Claims')
plt.title('Top 10 Providers by Claim Count')
plt.show()
```

### Feature Engineering & Suspicious Pattern Detection


##### 1. Identify High-Value Claims

```python
high_claim_threshold = merged_df['AMOUNT'].quantile(0.99)
suspicious_claims = merged_df[merged_df['AMOUNT'] > high_claim_threshold]
```

##### 2. Detect Repetitive & Rapid Claims
```python
merged_df['DATE'] = merged_df['SERVICEDATE'].dt.date
claims_per_provider_day = claims_df.groupby(['PROVIDERID', 'DATE']).size()
suspicious_frequency = claims_per_provider_day[claims_per_provider_day > 10]
```
##### 3. Anomaly Detection with Isolation Forest

```python
from sklearn.ensemble import IsolationForest
import numpy as np

# Prepare features
features = merged_df[['AMOUNT']]
features.fillna(0, inplace=True

clf = IsolationForest(contamination=0.01, random_state=42)
merged_df['anomaly_score'] = clf.fit_predict(features)

# Filter suspected fraudulent claims
suspects = merged_df[merged_df['anomaly_score'] == -1]
```
#### Visualization of Suspicious Activities

```python
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

### Concepts and Techniques Used

> **Outlier Detection Techniques:**  Isolation Forests identify anomalies based on isolation in feature space.

> **Feature Engineering:** Claim amounts, claim frequency, and temporal patterns.

> **Visualization:** Box plots, histograms, network graphs to visualize suspicious activity.

> **Thresholding:** Using statistical thresholds (e.g., 99th percentile) to flag high claims.

> **Network Analysis:** Exploring relationships between providers and patients.

### Findings & Next Steps
> High claim amounts and frequent claims flagged as potential fraud.

> Outlier claims identified via anomaly detection.

> Network analysis reveals clusters of providers and patients with suspicious interactions.

> Next steps include integrating supervised learning models with labeled data, refining feature sets, and developing dashboards for ongoing monitoring.

### Final Notes
> This analysis serves as an initial screening tool.

> False positives are possible; manual review is essential.

> Models should be continuously updated with new data and feedback.


## References & Resources
- **Synthea:** Synthetic Healthcare Data Generation
    - Main GitHub repo: https://github.com/synthetichealth/synthea
    - International extension: https://github.com/synthetichealth/synthea-international
- Generating Data for Ireland (IE):
    - Synthea can be customized for Ireland’s healthcare systems, demographics, and coding standards.
- See **README.md** on respective repo for for setup instructions.
- Documentation & Code: Synthea documentation: https://github.com/synthetichealth/synthea/wiki

## Authors & Acknowledgments
- Nithin Mohan T K- [@nithinmohantk]
- Inspiration from [Research Paper - Graph Analysis for Detecting Fraud, Waste, and Abuse in Healthcare Data] (https://ojs.aaai.org/aimagazine/index.php/aimagazine/article/view/2630/2554) by Juan Liu, Eric Bier, Aaron Wilson, Tomo Honda, Sricharan Kumar,
Leilani Gilpin, John Guerra-Gomez and Daniel Davies - Palo Alto Research Center
- Contributions from David Mullins, Anna Coyle & Dovile Janusauskaite 

## License
This project is licensed under the PRIVATE & COPYRIGHTED License. See the LICENSE file for details.

**© Neural Nexus Team** | Evernorth — All rights reserved.

With love for healthcare data analysis and fraud detection.
