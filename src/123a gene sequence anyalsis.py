import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

#Fix file path if needed
df = pd.read_csv(
    'GDS4758.soft.gz',
    skiprows=141,
    sep='\t',
    skipfooter=1,
    engine='python'         # ← enable skipfooter
)


# Manually enetered where and status of the sample from the .soft file
values = [
    "AD_HI", "AD_HI", "AD_HI", "AD_HI", "AD_HI", "AD_HI", "AD_HI",
    "AD_TC", "AD_TC", "AD_TC", "AD_TC", "AD_TC", "AD_TC", "AD_TC", "AD_TC", "AD_TC", "AD_TC",
    "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC",
    "AD_FC", "AD_FC", "AD_FC", "AD_FC", "AD_FC",
    "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI", "non-AD_HI",
    "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC",
    "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC", "non-AD_TC",
    "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC",
    "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC", "non-AD_FC"
]

#Format the new status/location row
new_row = ["AD_Status/Location", "AD_Status/Location"] + values

new_row += [None] * (len(df.columns) - len(new_row))

df.loc[len(df)] = new_row

# Clean the data for this df
ad_status = df.iloc[-1]          # <--- keep this as strings for later
df = df.drop(df.index[-1])

df.set_index("ID_REF", inplace=True)

df = df.apply(pd.to_numeric, errors='coerce')

# Calculate variance for each gene
gene_variance = df.var(axis=1)

# Select top 50/800 genes based on variance
top_50_genes = gene_variance.nlargest(50).index
df_top_genes = df.loc[top_50_genes]

top_800_genes = gene_variance.nlargest(800).index
df_top800_genes = df.loc[top_800_genes]

# Visualizations
# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_top_genes, cmap="viridis", xticklabels=10, yticklabels=10)
plt.title('Top Genes with Highest Variance Heatmap')
plt.xlabel('Sample')
plt.xticks(rotation=45)
plt.ylabel('Gene Identifier')
plt.show()

# Extract AD status
ad_strings = ad_status                                 # <--- preserve originals
ad = ad_strings.str.split("_").str[0]
ad_status = ad.map(lambda x: 1 if x == "AD" else 0)
ad_status = ad_status.drop(columns=['IDENTIFIER'])
ad_status = ad_status.iloc[2:]

# Drop AD status row from df_top_genes
df_top_genes = df_top800_genes.drop(df_top800_genes.index[-1])
df_top_genes = df_top_genes.drop(columns=['IDENTIFIER'])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(ad_status)
X = df_top_genes.T
# Random Forest Classifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)
rf_classifier.fit(X_train, y_train)

y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.3f}")

print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
feature_importance = rf_classifier.feature_importances_
top_genes = X.columns
important_features = pd.Series(feature_importance, index=top_genes).sort_values(ascending=False)

print("Top 15 Important Genes based on Random Forest:")
print(important_features.head(15))

svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
svm_classifier.fit(X_train, y_train)

# Predict
y_pred = svm_classifier.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Accuracy: {accuracy:.3f}")
print("Classification Report:\n", classification_report(y_test, y_pred))




