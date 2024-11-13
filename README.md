Here's the code breakdown and explanation formatted for a README file. You can copy and paste this directly into your README to explain each part of the Yelp Review Classification project.

---

# Yelp Review Classification Project

This project uses Natural Language Processing (NLP) and machine learning to classify Yelp reviews as either 1-star (negative) or 5-star (positive) based on the review text. The project leverages Python, Scikit-Learn, and Logistic Regression for text classification.

## Project Workflow

1. **Load and Preprocess Data**: Load Yelp review data and filter it to include only 1-star and 5-star reviews for binary classification.
2. **Train-Test Split**: Split the dataset into training and testing sets.
3. **Build a Pipeline**: Create a Scikit-Learn pipeline to preprocess text data and train a Logistic Regression model.
4. **Evaluate the Model**: Generate evaluation metrics including a confusion matrix, classification report, learning curve, cross-validation scores, and ROC-AUC curve.

---

## Code Breakdown

### Step 1: Import Necessary Libraries

```python
# Import libraries for data manipulation, visualization, and machine learning
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.preprocessing import LabelBinarizer
```

*Explanation*: This section imports libraries for data handling (Pandas), visualization (Matplotlib and Seaborn), and text processing, model training, and evaluation (Scikit-Learn).

---

### Step 2: Load and Preprocess Data

```python
# Load the Yelp data
yelp = pd.read_csv('D:/Placements/key academic projects/Optimization Algorithms in ML/yelp.csv')

# Filter the dataset to include only 1-star and 5-star reviews for binary classification
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]
X = yelp_class['text']  # Feature: Review text
y = yelp_class['stars'] # Target: Star rating (1 or 5)
```

*Explanation*: This step loads the Yelp dataset and filters it to include only reviews with 1-star and 5-star ratings. The `X` variable contains the review text (features), and `y` contains the corresponding ratings (target).

---

### Step 3: Train-Test Split

```python
# Split the data into training and testing sets (67% train, 33% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)
```

*Explanation*: The dataset is split into training and testing sets, with 67% of the data used for training and 33% for testing. A fixed random state ensures reproducibility.

---

### Step 4: Create a Pipeline for Data Processing and Model Training

```python
# Define a pipeline with CountVectorizer, TfidfTransformer, and Logistic Regression
pipe = Pipeline([
    ('bow', CountVectorizer()),             # Step 1: Tokenization and Count Vectorization
    ('tfidf', TfidfTransformer()),          # Step 2: TF-IDF transformation
    ('model', LogisticRegression())         # Step 3: Logistic Regression model
])
```

*Explanation*: The pipeline automates the text processing and model training steps:
- **CountVectorizer**: Converts review text into a matrix of token counts.
- **TfidfTransformer**: Transforms token counts into TF-IDF features.
- **LogisticRegression**: The classifier used to predict whether a review is 1-star or 5-star.

---

### Step 5: Train the Model

```python
# Fit the model to the training data
pipe.fit(X_train, y_train)
```

*Explanation*: The pipeline is trained on the training data, performing text preprocessing and model training in one step.

---

### Step 6: Make Predictions

```python
# Make predictions on the test data
predictions = pipe.predict(X_test)
```

*Explanation*: This step uses the trained pipeline to predict star ratings for the test set reviews.

---

### Step 7: Evaluate the Model

#### 7.1 Confusion Matrix

```python
# Generate a confusion matrix
cm = confusion_matrix(y_test, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 5], yticklabels=[1, 5])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
```

*Explanation*: The confusion matrix visualizes the number of correct and incorrect predictions. A heatmap displays the matrix, showing true positives, false positives, true negatives, and false negatives.

#### 7.2 Classification Report

```python
# Display a classification report
print("Classification Report:\n", classification_report(y_test, predictions))
```

*Explanation*: The classification report provides precision, recall, F1-score, and support for each class, helping to understand model performance on positive and negative reviews.

#### 7.3 Learning Curve

```python
# Plot the learning curve
train_sizes, train_scores, test_scores = learning_curve(
    pipe, X_train, y_train, cv=5, scoring='accuracy', 
    train_sizes=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
)

# Calculate the mean and standard deviation of training and test scores
train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
test_std = test_scores.std(axis=1)

# Plot learning curve
plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_mean, label="Training score", color='blue')
plt.plot(train_sizes, test_mean, label="Cross-validation score", color='green')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color='blue')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.2, color='green')
plt.title('Learning Curve')
plt.xlabel('Training Size')
plt.ylabel('Accuracy')
plt.legend(loc='best')
plt.show()
```

*Explanation*: The learning curve helps evaluate model performance as more training data is added, indicating whether the model is overfitting or underfitting.

#### 7.4 Cross-Validation Scores

```python
# Perform cross-validation
cv_scores = cross_val_score(pipe, X, y, cv=5, scoring='accuracy')

# Display cross-validation scores
print(f'Cross-validation scores: {cv_scores}')
print(f'Mean Cross-validation score: {cv_scores.mean():.4f}')
```

*Explanation*: Cross-validation scores give an estimate of the model's performance across different subsets of the data, helping to understand its generalizability.

#### 7.5 ROC Curve and AUC

```python
# Binarize output labels for ROC curve
lb = LabelBinarizer()
y_bin_test = lb.fit_transform(y_test)

# Get predicted probabilities for positive class
y_pred_prob = pipe.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_bin_test, y_pred_prob)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc='best')
plt.show()

# Calculate AUC score
roc_auc = roc_auc_score(y_bin_test, y_pred_prob)
print(f'Area Under Curve (AUC): {roc_auc:.4f}')
```

*Explanation*: The ROC curve shows the model's performance in distinguishing between classes across different thresholds, and the AUC score quantifies the model's ability to separate positive and negative reviews.

---

By following these steps, you can reproduce this project and understand the workflow of building a text classification model using logistic regression and Scikit-Learn. This breakdown explains each component's role in the model-building process.
