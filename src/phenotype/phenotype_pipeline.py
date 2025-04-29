import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

#OASIS dataset sector of loading
oasis_df = pd.read_csv('oasis_cross-sectional.csv').dropna(subset=['CDR'])
oasis_df['Demented'] = (oasis_df['CDR'] > 0).astype(int)
#Here we are setting True if CDR>0

for col in ['Educ', 'SES', 'MMSE']:
  # remember to inpute missing values
    oasis_df[col].fillna(oasis_df[col].median(), inplace=True)

# Encode categorical features
oasis_df['M_F']    = LabelEncoder().fit_transform(oasis_df['M/F'])
oasis_df['Handed'] = LabelEncoder().fit_transform(oasis_df['Hand'])
#Similarly for Handed, turning “L”/“R” into 0/1



#dementia prevalence (donut chart)
sns.set(style='whitegrid')
plt.figure(figsize=(5,5))
oasis_df['Demented'].value_counts().plot.pie(
  #counting how many 0s vs 1s, then make a pie chart with that series.

    labels=['No Dementia','Dementia'],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'width':0.4,'edgecolor':'w'},
    colors=sns.color_palette('pastel')[0:2]
)
plt.title('Dementia Prevalence')
plt.ylabel('')
plt.savefig('oasis_dementia_donut.png', dpi=150)
plt.show()

#Age vs. MMSE scatter plot
plt.figure(figsize=(7,5))
sns.scatterplot(
    x='Age', y='MMSE', hue='Demented',
    data=oasis_df,
    palette=['skyblue','salmon'], alpha=0.7, s=80
)
plt.title('Age vs. MMSE')
plt.xlabel('Age (years)')
plt.ylabel('MMSE Score')
plt.legend(title='Demented')
plt.tight_layout()
plt.savefig('age_vs_mmse.png', dpi=150)
plt.show()


features = ['M_F','Handed','Age','Educ','SES','MMSE','eTIV','nWBV','ASF']
X = oasis_df[features]

y = oasis_df['Demented']
# here is the prepping data for modeling

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
) #keep the same dementia/no-dementia ratio in both sets.

# Train Decision Tree
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

# visual and save the tree
plt.figure(figsize=(10,6))
plot_tree(
    dt,
    feature_names=features,
    class_names=['No Dementia','Dementia'],
    filled=True, rounded=True, fontsize=8
)
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=150)
plt.show()

# Evaluate model ML
y_pred = dt.predict(X_test)
print(f"Decision Tree Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred, zero_division=0))

