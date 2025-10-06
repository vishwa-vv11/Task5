# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
# Make sure your CSV file is named 'heart.csv' and contains the columns as in the image you shared
df = pd.read_csv('heart.csv')

# Split the data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 1. Train a Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=["No Disease", "Disease"], filled=True)
plt.title("Decision Tree")
plt.show()

# Evaluate the model
train_acc = dt.score(X_train, y_train)
test_acc = dt.score(X_test, y_test)
print(f"Decision Tree - Training Accuracy: {train_acc:.2f}")
print(f"Decision Tree - Test Accuracy: {test_acc:.2f}")

# 2. Analyze overfitting by limiting depth
for depth in range(1, 11):
    dt_limited = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt_limited.fit(X_train, y_train)
    train_score = dt_limited.score(X_train, y_train)
    test_score = dt_limited.score(X_test, y_test)
    print(f"Depth: {depth}, Train Acc: {train_score:.2f}, Test Acc: {test_score:.2f}")

# 3. Train a Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"Random Forest - Test Accuracy: {rf_acc:.2f}")

# 4. Interpret feature importances
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp, y=feat_imp.index)
plt.title("Feature Importances - Random Forest")
plt.show()

# 5. Evaluate using cross-validation
cv_scores_dt = cross_val_score(dt, X, y, cv=5)
cv_scores_rf = cross_val_score(rf, X, y, cv=5)
print(f"Decision Tree CV Accuracy: {cv_scores_dt.mean():.2f}")
print(f"Random Forest CV Accuracy: {cv_scores_rf.mean():.2f}")
