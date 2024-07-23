import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
from wordcloud import WordCloud
from textblob import TextBlob
import string
import matplotlib.backends.backend_pdf

warnings.filterwarnings("ignore")

# Phase 1: Data Management

# Step 1: Read datasets
print("Reading datasets...")
profiles = pd.read_csv('C:/Users/Admin/Documents/Data Science/DATASETS/customer profiles.csv')
info = pd.read_csv('C:/Users/Admin/Documents/Data Science/DATASETS/customer info.csv')
details = pd.read_csv('C:/Users/Admin/Documents/Data Science/DATASETS/customer credit details.csv')

# Step 2: Merge datasets to create master data
print("Merging datasets...")
master_data = pd.merge(details, info, on='custid')
master_data = pd.merge(master_data, profiles, on='custid')

# Step 3: Data Understanding
print("Understanding the master data...")
print("Shape of master data:", master_data.shape)
print("Columns in master data:", master_data.columns)
print("Data types of each column:\n", master_data.dtypes)
print("Missing values in each column:\n", master_data.isnull().sum())
print("Number of unique values in each column:\n", master_data.nunique())
print("Summary statistics of master data:\n", master_data.describe())

# Step 4: Handling duplicates
print("Handling duplicates...")
master_data = master_data.drop_duplicates(subset='custid')
print("Summary statistics of master data:\n", master_data.describe())

# Step 5: Data Cleaning
print("Cleaning the data...")
master_data.replace('', np.nan, inplace=True)
master_data['othdebt'] = master_data['othdebt'].replace(' ', np.nan)
master_data.dropna(inplace=True)

# Correcting specific values in the categorical variables
print("Correcting specific values in categorical variables...")
master_data['deposit'] = master_data['deposit'].replace({11: 1, 22: 2})
master_data['selfemp'] = master_data['selfemp'].replace({11: 1, 22: 2})
master_data['veh'] = master_data['veh'].replace({11: 1, 22: 2})
master_data['preloan'] = master_data['preloan'].replace({'11': '1'})

# Converting columns to appropriate data types
print("Converting columns to appropriate data types...")
categorical_columns = ['age', 'veh', 'house', 'selfemp', 'deposit', 'branch', 'ref', 'gender', 'ms', 'child', 'preloan']
for col in categorical_columns:
    master_data[col] = master_data[col].astype('category')
master_data['bad'] = master_data['bad'].astype('int')
master_data['othdebt'] = master_data['othdebt'].astype('float')

# Converting continuous variables into categorical variables using bins
print("Converting continuous variables into categorical variables...")
debtinc_bins = [0, 7, 11, 16, 46]
creddebt_bins = [0, 1, 2, 3, 23]
othdebt_bins = [0, 2, 3, 5, 29]
emp_bins = [0, 3, 8, 13, 33]

master_data['debtinc_ctg'] = pd.cut(master_data['debtinc'], bins=debtinc_bins, labels=['(0-7)', '(7-11)', '(11-16)', '(16-46)'])
master_data['creddebt_ctg'] = pd.cut(master_data['creddebt'], bins=creddebt_bins, labels=['(0-1)', '(1-2)', '(2-3)', '(3-23)'])
master_data['othdebt_ctg'] = pd.cut(master_data['othdebt'], bins=othdebt_bins, labels=['(0-2)', '(2-3)', '(3-5)', '(5-29)'])
master_data['emp_ctg'] = pd.cut(master_data['emp'], bins=emp_bins, labels=['(0-3)', '(3-8)', '(8-13)', '(13-33)'], include_lowest=True, right=False)

# Removing custid from the dataset as it is an identifier
master_data = master_data.drop(columns=['custid'])

print("Data types of each column:\n", master_data.dtypes)

# Finding overall “Bad Rate/Defaulter Rate” in the data
bad_rate = master_data['bad'].mean() * 100
print(f"Overall Defaulter Rate: {bad_rate:.2f}%")

# Phase 2: Data Analysis using Graphs, Tables, etc. (EDA)

# Box-Whisker plot for numeric variables
plt.figure(figsize=(6, 8))
sns.boxplot(x='bad', y='debtinc', data=master_data, palette='pastel')
plt.title('Debt-to-Income Ratio by Bad')
plt.show()

plt.figure(figsize=(6, 8))
sns.boxplot(x='bad', y='creddebt', data=master_data, palette='pastel')
plt.title('Credit Debt by Bad')
plt.show()

plt.figure(figsize=(6, 8))
sns.boxplot(x='bad', y='othdebt', data=master_data, palette='pastel')
plt.title('Other Debt by Bad')
plt.show()

# Summary statistics for numeric variables based on 'bad'
debtinc_summary = master_data.groupby('bad')['debtinc'].describe()
creddebt_summary = master_data.groupby('bad')['creddebt'].describe()
othdebt_summary = master_data.groupby('bad')['othdebt'].describe()

print("Debt-to-Income Ratio Summary by Bad:")
print(debtinc_summary)

print("\nCredit Debt Summary by Bad:")
print(creddebt_summary)

print("\nOther Debt Summary by Bad:")
print(othdebt_summary)

# Table of “Bad Rate” for categorical variables
print("Table of 'Bad Rate' for categorical variables:")
gender_bad_rate = pd.crosstab(master_data['gender'], master_data['bad'], normalize='index') * 100
print(gender_bad_rate.round(2))

zone_bad_rate = pd.crosstab(master_data['zone'], master_data['bad'], normalize='index') * 100
print(zone_bad_rate.round(2))

# Bar Chart: Mean of numeric variables for Bad=YES/NO
mean_by_bad = master_data.groupby('bad')[['debtinc', 'creddebt', 'othdebt', 'emp']].mean()
mean_by_bad.plot(kind='bar')
plt.title('Mean of Numeric Variables by Bad')
plt.xlabel('Bad')
plt.ylabel('Mean')
plt.show()

print(mean_by_bad)

# Heatmap for 2 variables showing “bad rate”
print("Generating heatmap for bad rate by zone and credit debt category...")
selected_vars = ['zone', 'creddebt_ctg', 'bad']
data_selected = master_data[selected_vars].dropna()

# Create the pivot table
pivot_table = data_selected.pivot_table(index='zone', columns='creddebt_ctg', values='bad', aggfunc='mean')

# Display the pivot table values
print("Pivot Table for Bad Rate by Zone and Credit Debt Category:")
print(pivot_table)

print("Generating heatmap for bad rate by zone and credit debt category...")
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(data_selected.pivot_table(index='zone', columns='creddebt_ctg', values='bad', aggfunc='mean'), annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Bad Rate by Zone and Credit Debt Category')
plt.xlabel('Credit Debt Category')
plt.ylabel('Zone')
plt.show()

# Phase 3: Developing a model using Binary Logistic Regression

# Step 1: Create data partition into train and test data sets (80/20)
print("Creating data partition into train and test sets...")
train_data, test_data = train_test_split(master_data, test_size=0.2, random_state=23)

# Step 2: Run Binary Logistic Regression with “bad” as dependent variable and all others as independent variables on train data
print("Running initial Binary Logistic Regression model...")
initial_model = smf.logit(formula='bad ~ age + gender + ms + child + zone + veh + house + selfemp + account + deposit + emp + address + branch + ref + debtinc + creddebt + othdebt + preloan', data=train_data).fit()
print(initial_model.summary())

# Step 3: Check which variables are significant (p-value < 0.05) and refine the model
print("Refining the model based on significant variables...")
significant_vars = ['branch', 'emp', 'address', 'debtinc', 'creddebt']
refined_model = smf.logit(formula='bad ~ ' + '+'.join(significant_vars), data=train_data).fit()
print(refined_model.summary())

# Step 4: Check for multicollinearity using VIF
print("Checking for multicollinearity using VIF...")
vif_features = train_data[significant_vars]
vif_features = sm.add_constant(vif_features)
vif_data = pd.DataFrame()
vif_data["Feature"] = vif_features.columns
vif_data["VIF"] = [variance_inflation_factor(vif_features.values, i) for i in range(vif_features.shape[1])]
print(vif_data)

# Step 5: Obtain ROC curve and AUC for train data
print("Obtaining ROC curve and AUC for train data...")
train_probabilities = refined_model.predict(train_data)
train_labels = train_data['bad']
fpr_train, tpr_train, thresholds_train = roc_curve(train_labels, train_probabilities)
auc_train = roc_auc_score(train_labels, train_probabilities)

plt.figure(figsize=(8, 6))
plt.plot(fpr_train, tpr_train, color='blue', label=f'ROC Curve (AUC = {auc_train:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Training Data')
plt.legend(loc='lower right')
plt.show()

# Step 6: Obtain confusion matrix and accuracy for train data
print("Obtaining confusion matrix and accuracy for train data...")
train_predictions = (train_probabilities > 0.5).astype(int)
train_conf_matrix = confusion_matrix(train_labels, train_predictions)
print("Confusion Matrix:")
print(train_conf_matrix)

train_accuracy = accuracy_score(train_labels, train_predictions)
print("\nAccuracy:", train_accuracy)
print(classification_report(train_labels, train_predictions))

# Step 7: Function to calculate metrics
def calculate_metrics(conf_matrix):
    TP = conf_matrix[1, 1]
    TN = conf_matrix[0, 0]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    misclassification_rate = (FP + FN) / conf_matrix.sum()
    
    return sensitivity, specificity, misclassification_rate

sensitivity, specificity, misclassification_rate = calculate_metrics(train_conf_matrix)
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Misclassification Rate: {misclassification_rate}")

# Step 8: Finding optimal threshold to balance sensitivity and specificity
optimal_threshold = 0.14
print(f"Optimal Threshold: {optimal_threshold}")

# Apply the optimal threshold to get balanced predictions
balanced_train_predictions = (train_probabilities > optimal_threshold).astype(int)
balanced_train_conf_matrix = confusion_matrix(train_labels, balanced_train_predictions)
balanced_train_accuracy = accuracy_score(train_labels, balanced_train_predictions)
print("Balanced Confusion Matrix:")
print(balanced_train_conf_matrix)
print("\nBalanced Accuracy:", balanced_train_accuracy)
print(classification_report(train_labels, balanced_train_predictions))

balanced_sensitivity, balanced_specificity, balanced_misclassification_rate = calculate_metrics(balanced_train_conf_matrix)
print(f"Sensitivity: {balanced_sensitivity}")
print(f"Specificity: {balanced_specificity}")
print(f"Misclassification Rate: {balanced_misclassification_rate}")

# Phase 4: Using ML methods and compare with Binary Logistic Regression

# Step 1: Apply Naive Bayes Method on train data
print("Applying Naive Bayes Method on train data...")
nb_model = GaussianNB()
nb_model.fit(train_data[significant_vars], train_labels)

# Step 2: Obtain ROC curve and AUC for train data (Naive Bayes)
print("Obtaining ROC curve and AUC for train data (Naive Bayes)...")
train_probs_nb = nb_model.predict_proba(train_data[significant_vars])[:, 1]
fpr_train_nb, tpr_train_nb, _ = roc_curve(train_labels, train_probs_nb)
auc_train_nb = roc_auc_score(train_labels, train_probs_nb)
print(f"Naive Bayes Train AUC: {auc_train_nb:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_train_nb, tpr_train_nb, color='blue', label=f'Naive Bayes ROC Curve (AUC = {auc_train_nb:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes (Train Data)')
plt.legend(loc='lower right')
plt.show()

# Step 3: Obtain Confusion Matrix for train data (Naive Bayes)
print("Obtaining Confusion Matrix for train data (Naive Bayes)...")
train_preds_nb = nb_model.predict(train_data[significant_vars])
conf_matrix_train_nb = confusion_matrix(train_labels, train_preds_nb)
print("Naive Bayes Train Confusion Matrix:")
print(conf_matrix_train_nb)

accuracy_train_nb = accuracy_score(train_labels, train_preds_nb)
print("\nNaive Bayes Train Accuracy:", accuracy_train_nb)
print(classification_report(train_labels, train_preds_nb))

# Step 4: Apply Decision Tree on train data
print("Applying Decision Tree on train data...")
max_depth = 3  # Limit the depth of the tree for better visualization
dt_model_limited = DecisionTreeClassifier(random_state=23, max_depth=max_depth)
dt_model_limited.fit(train_data[significant_vars], train_labels)

# Plot the tree with limited depth
plt.figure(figsize=(16, 10))
plot_tree(dt_model_limited, filled=True, feature_names=significant_vars)
plt.title('Decision Tree Visualization (Limited Depth)')
plt.show()

# Export the tree to a PDF for better readability
pdf = matplotlib.backends.backend_pdf.PdfPages("decision_tree.pdf")
plt.figure(figsize=(20, 20))
plot_tree(dt_model_limited, filled=True, feature_names=significant_vars)
plt.title('Decision Tree Visualization')
pdf.savefig()
pdf.close()

# Step 5: Obtain ROC curve and AUC for train data (Decision Tree)
print("Obtaining ROC curve and AUC for train data (Decision Tree)...")
train_probs_dt = dt_model_limited.predict_proba(train_data[significant_vars])[:, 1]
fpr_train_dt, tpr_train_dt, _ = roc_curve(train_labels, train_probs_dt)
auc_train_dt = roc_auc_score(train_labels, train_probs_dt)
print(f"Decision Tree Train AUC: {auc_train_dt:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_train_dt, tpr_train_dt, color='blue', label=f'Decision Tree ROC Curve (AUC = {auc_train_dt:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree (Train Data)')
plt.legend(loc='lower right')
plt.show()

# Step 6: Obtain Confusion Matrix for train data (Decision Tree)
print("Obtaining Confusion Matrix for train data (Decision Tree)...")
train_preds_dt = dt_model_limited.predict(train_data[significant_vars])
conf_matrix_train_dt = confusion_matrix(train_labels, train_preds_dt)
print("Decision Tree Train Confusion Matrix:")
print(conf_matrix_train_dt)

accuracy_train_dt = accuracy_score(train_labels, train_preds_dt)
print("\nDecision Tree Train Accuracy:", accuracy_train_dt)
print(classification_report(train_labels, train_preds_dt))

# Step 7: Apply Random Forest Method on train data
print("Applying Random Forest Method on train data...")
rf_model = RandomForestClassifier(random_state=23)
rf_model.fit(train_data[significant_vars], train_labels)

# Step 8: Obtain ROC curve and AUC for train data (Random Forest)
print("Obtaining ROC curve and AUC for train data (Random Forest)...")
train_probs_rf = rf_model.predict_proba(train_data[significant_vars])[:, 1]
fpr_train_rf, tpr_train_rf, _ = roc_curve(train_labels, train_probs_rf)
auc_train_rf = roc_auc_score(train_labels, train_probs_rf)
print(f"Random Forest Train AUC: {auc_train_rf:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_train_rf, tpr_train_rf, color='blue', label=f'Random Forest ROC Curve (AUC = {auc_train_rf:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest (Train Data)')
plt.legend(loc='lower right')
plt.show()

# Step 9: Obtain Confusion Matrix for train data (Random Forest)
print("Obtaining Confusion Matrix for train data (Random Forest)...")
train_preds_rf = rf_model.predict(train_data[significant_vars])
conf_matrix_train_rf = confusion_matrix(train_labels, train_preds_rf)
print("Random Forest Train Confusion Matrix:")
print(conf_matrix_train_rf)

accuracy_train_rf = accuracy_score(train_labels, train_preds_rf)
print("\nRandom Forest Train Accuracy:", accuracy_train_rf)
print(classification_report(train_labels, train_preds_rf))

# Phase 4: Test Data Evaluation and Model Comparison

# Logistic Regression on Test Data
print("Evaluating Logistic Regression on Test Data...")
test_probabilities = refined_model.predict(test_data)
test_labels = test_data['bad']
fpr_test, tpr_test, _ = roc_curve(test_labels, test_probabilities)
auc_test = roc_auc_score(test_labels, test_probabilities)
print(f"Logistic Regression Test AUC: {auc_test:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_test, tpr_test, color='blue', label=f'Logistic Regression ROC Curve (AUC = {auc_test:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Logistic Regression (Test Data)')
plt.legend(loc='lower right')
plt.show()

test_preds = (test_probabilities > optimal_threshold).astype(int)
conf_matrix_test = confusion_matrix(test_labels, test_preds)
print("Logistic Regression Test Confusion Matrix:")
print(conf_matrix_test)

accuracy_test = accuracy_score(test_labels, test_preds)
print("\nLogistic Regression Test Accuracy:", accuracy_test)
print(classification_report(test_labels, test_preds))

# Naive Bayes on Test Data
print("Evaluating Naive Bayes on Test Data...")
test_probs_nb = nb_model.predict_proba(test_data[significant_vars])[:, 1]
fpr_test_nb, tpr_test_nb, _ = roc_curve(test_labels, test_probs_nb)
auc_test_nb = roc_auc_score(test_labels, test_probs_nb)
print(f"Naive Bayes Test AUC: {auc_test_nb:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_test_nb, tpr_test_nb, color='blue', label=f'Naive Bayes ROC Curve (AUC = {auc_test_nb:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Naive Bayes (Test Data)')
plt.legend(loc='lower right')
plt.show()

test_preds_nb = (test_probs_nb > optimal_threshold).astype(int)
conf_matrix_test_nb = confusion_matrix(test_labels, test_preds_nb)
print("Naive Bayes Test Confusion Matrix:")
print(conf_matrix_test_nb)

accuracy_test_nb = accuracy_score(test_labels, test_preds_nb)
print("\nNaive Bayes Test Accuracy:", accuracy_test_nb)
print(classification_report(test_labels, test_preds_nb))

# Decision Tree on Test Data
print("Evaluating Decision Tree on Test Data...")
test_probs_dt = dt_model_limited.predict_proba(test_data[significant_vars])[:, 1]
fpr_test_dt, tpr_test_dt, _ = roc_curve(test_labels, test_probs_dt)
auc_test_dt = roc_auc_score(test_labels, test_probs_dt)
print(f"Decision Tree Test AUC: {auc_test_dt:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_test_dt, tpr_test_dt, color='blue', label=f'Decision Tree ROC Curve (AUC = {auc_test_dt:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Decision Tree (Test Data)')
plt.legend(loc='lower right')
plt.show()

test_preds_dt = (test_probs_dt > optimal_threshold).astype(int)
conf_matrix_test_dt = confusion_matrix(test_labels, test_preds_dt)
print("Decision Tree Test Confusion Matrix:")
print(conf_matrix_test_dt)

accuracy_test_dt = accuracy_score(test_labels, test_preds_dt)
print("\nDecision Tree Test Accuracy:", accuracy_test_dt)
print(classification_report(test_labels, test_preds_dt))

# Random Forest on Test Data
print("Evaluating Random Forest on Test Data...")
test_probs_rf = rf_model.predict_proba(test_data[significant_vars])[:, 1]
fpr_test_rf, tpr_test_rf, _ = roc_curve(test_labels, test_probs_rf)
auc_test_rf = roc_auc_score(test_labels, test_probs_rf)
print(f"Random Forest Test AUC: {auc_test_rf:.2f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr_test_rf, tpr_test_rf, color='blue', label=f'Random Forest ROC Curve (AUC = {auc_test_rf:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Random Forest (Test Data)')
plt.legend(loc='lower right')
plt.show()

test_preds_rf = (test_probs_rf > optimal_threshold).astype(int)
conf_matrix_test_rf = confusion_matrix(test_labels, test_preds_rf)
print("Random Forest Test Confusion Matrix:")
print(conf_matrix_test_rf)

accuracy_test_rf = accuracy_score(test_labels, test_preds_rf)
print("\nRandom Forest Test Accuracy:", accuracy_test_rf)
print(classification_report(test_labels, test_preds_rf))

# Model Comparison on Test Data
print("Comparing models on test data...")
model_comparison = pd.DataFrame({
    'Model': ['Logistic Regression', 'Naive Bayes', 'Decision Tree', 'Random Forest'],
    'Accuracy': [accuracy_test, accuracy_test_nb, accuracy_test_dt, accuracy_test_rf],
    'AUC': [auc_test, auc_test_nb, auc_test_dt, auc_test_rf]
})

print("Model Comparison on Test Data:")
print(model_comparison)

# Phase 5: Analyzing Customer Feedback Data

# Step 1: Read the customer feedback data
feedback_file_path = 'C:/Users/Admin/Documents/Data Science/DATASETS/Customer Feedback.txt'
feedback_data = pd.read_csv(feedback_file_path, delimiter='\t', header=None, names=['Feedback'])

# Step 2: Text Pre-processing
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Remove numbers
    text = ''.join([char for char in text if not char.isdigit()])
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

feedback_data['Processed_Feedback'] = feedback_data['Feedback'].apply(preprocess_text)

# Step 3: Generate Word Cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(feedback_data['Processed_Feedback']))
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Customer Feedback Word Cloud')
plt.show()

# Get top 10 most mentioned words
vectorizer = CountVectorizer(stop_words='english')
word_counts = vectorizer.fit_transform(feedback_data['Processed_Feedback'])
word_freq = pd.DataFrame(word_counts.toarray(), columns=vectorizer.get_feature_names_out()).sum().sort_values(ascending=False).head(10)
print("Top 10 Most Mentioned Words:")
print(word_freq)

# Bar Chart of top 10 most mentioned words
plt.figure(figsize=(10, 6))
sns.barplot(x=word_freq.values, y=word_freq.index, palette='pastel')
plt.title('Top 10 Most Mentioned Words')
plt.xlabel('Frequency')
plt.ylabel('Words')
plt.show()

# Pie Chart of top 5 most mentioned words
top_5_words = word_freq.head(5)
plt.figure(figsize=(8, 8))
plt.pie(top_5_words, labels=top_5_words.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
plt.title('Top 5 Most Mentioned Words')
plt.show()

# Step 4: Sentiment Analysis
def get_sentiment(text):
    analysis = TextBlob(text)
    return analysis.sentiment.polarity

feedback_data['Sentiment_Score'] = feedback_data['Processed_Feedback'].apply(get_sentiment)
feedback_data['Sentiment'] = feedback_data['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))

# Step 5: Visualize Sentiment Scores
plt.figure(figsize=(8, 6))
sns.histplot(feedback_data['Sentiment_Score'], bins=20, kde=True, color='blue')
plt.title('Distribution of Sentiment Scores')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()

# Step 6: Sentiment Analysis Summary
sentiment_summary = feedback_data['Sentiment'].value_counts(normalize=True) * 100
print("Sentiment Analysis Summary (%):")
print(sentiment_summary)

# Step 7: Plot Sentiment Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Sentiment', data=feedback_data, palette='pastel')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()

# Step 8: Summary statistics of sentiment scores
sentiment_summary = feedback_data['Sentiment_Score'].describe()
print("Summary Statistics of Sentiment Scores:")
print(sentiment_summary)

# Step 9: Distribution counts of sentiment categories
feedback_data['Sentiment'] = feedback_data['Sentiment_Score'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
sentiment_counts = feedback_data['Sentiment'].value_counts(normalize=True) * 100
print("Sentiment Distribution (%):")
print(sentiment_counts)
