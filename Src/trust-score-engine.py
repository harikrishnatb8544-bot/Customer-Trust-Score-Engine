import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


# 1. LOAD DATA

df = pd.read_csv("Data/26k-consumer-complaints.csv")


# 2. CLEAN COLUMN NAMES

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_")
    .str.replace("-", "_")
)

# 3. CREATE CUSTOMER ID

df['customer_key'] = (
    df['company'].astype(str) + "_" +
    df['product'].astype(str) + "_" +
    df['issue'].astype(str) + "_" +
    df['sub_issue'].astype(str)
)

df['customer_id'] = df['customer_key'].astype('category').cat.codes


# 4. DATE HANDLING

df['date_received'] = pd.to_datetime(df['date_received'], dayfirst=True)


# 5. CUSTOMER-LEVEL AGGREGATION

customer_df = df.groupby('customer_id').agg(
    total_complaints=('complaint_id', 'count'),
    first_complaint=('date_received', 'min'),
    last_complaint=('date_received', 'max')
).reset_index()

customer_df['complaint_duration_days'] = (
    customer_df['last_complaint'] - customer_df['first_complaint']
).dt.days + 1

customer_df['complaint_frequency'] = (
    customer_df['total_complaints'] / customer_df['complaint_duration_days']
)


# 6. DISPUTE SIGNAL

df['consumer_disputed'] = df['consumer_disputed?'].map({'Yes': 1, 'No': 0}).fillna(0)

dispute_df = df.groupby('customer_id').agg(
    dispute_count=('consumer_disputed', 'sum')
).reset_index()

customer_df = customer_df.merge(dispute_df, on='customer_id', how='left')


# 7. COMPANY RESPONSE IMPACT

response_impact = {
    'Closed with monetary relief': 10,
    'Closed with non-monetary relief': 5,
    'Closed with explanation': -5,
    'In progress': -10,
    'Closed': -10,
    'Untimely response': -20
}

df['response_score'] = df['company_response'].map(response_impact).fillna(0)

response_df = df.groupby('customer_id').agg(
    response_score_total=('response_score', 'sum')
).reset_index()

customer_df = customer_df.merge(response_df, on='customer_id', how='left')
customer_df['response_score_total'] = customer_df['response_score_total'].fillna(0)


# 8. RULE-BASED TRUST SCORE

customer_df['trust_score'] = (
    100
    - customer_df['total_complaints'] * 5
    - customer_df['dispute_count'] * 15
    - (customer_df['complaint_frequency'] > 0.05).astype(int) * 10
    + customer_df['response_score_total']
)

customer_df['trust_score'] = customer_df['trust_score'].clip(0, 100)


# 9. ML LABEL
# 1 = High Trust | 0 = At Risk
customer_df['trust_label'] = customer_df['trust_score'].apply(
    lambda x: 1 if x >= 70 else 0
)


# 10. FEATURE SET

features = [
    'total_complaints',
    'complaint_frequency',
    'dispute_count',
    'response_score_total'
]

X = customer_df[features]
y = customer_df['trust_label']


# 11. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 12. TRAIN ML MODEL

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# 13. EVALUATION

y_pred = model.predict(X_test)

print("CONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))


# 14. PROBABILITY-BASED RISK OUTPUT

proba = model.predict_proba(customer_df[features])

customer_df['churn_risk_percent'] = proba[:, 0] * 100
customer_df['trust_percent'] = proba[:, 1] * 100


# 15. BUSINESS RISK BUCKETS

def risk_category(risk):
    if risk >= 80:
        return 'High Churn Risk'
    elif risk >= 50:
        return 'Moderate Risk'
    else:
        return 'High Trust Customer'

customer_df['customer_status'] = customer_df['churn_risk_percent'].apply(risk_category)


# 16. FINAL OUTPUT

final_output = customer_df[
    [
        'customer_id',
        'churn_risk_percent',
        'trust_percent',
        'customer_status'
    ]
]

print("\nFINAL CUSTOMER RISK OUTPUT:")
print(final_output.head(10))
