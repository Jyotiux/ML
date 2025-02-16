## MNISTdatasetclassifier.ipynb

Dataset & Preprocessing:

Loaded the MNIST dataset (handwritten digits).
Split the data into training and test sets.
Applied MinMaxScaler for feature scaling.
Used PCA (Principal Component Analysis) for dimensionality reduction (64 components).
K-Nearest Neighbors (KNN):

Used GridSearchCV to tune hyperparameters like n_neighbors, weights, and metric.
Best parameters found: n_neighbors=3, weights=distance, metric=euclidean.
Achieved Test Accuracy: 96.24%
Random Forest Classifier:

Used GridSearchCV to tune n_estimators, max_depth, and min_samples_split.
Best parameters found: n_estimators=150, max_depth=20, min_samples_split=5.
Achieved Test Accuracy: 92.85%
Support Vector Machine (SVM):

Used GridSearchCV to tune C, gamma, and kernel.
Best parameters found: C=5, gamma=scale, kernel=rbf.
Achieved Test Accuracy: 95.73%
Ensemble Methods:

Voting Classifier: Combined KNN, Random Forest, and SVM using hard voting.
Achieved Test Accuracy: 95.98%
Stacking Classifier: Used KNN, Random Forest, and SVM with Logistic Regression as the meta-learner.
Achieved Test Accuracy: 95.73%
Encountered a Convergence Warning (suggesting increasing max_iter or scaling improvements).
Convolutional Neural Network (CNN) Model:

Built a simple CNN with:
Conv2D + MaxPooling2D + Flatten + Dense layers.
Trained the model for 10 epochs.
Achieved Test Accuracy: 98.52%

## Titanicdataset.ipynb

Data Loading & Exploration

Loaded train.csv and test.csv using Pandas.
Identified missing values in Age, Cabin, and Embarked attributes.
Explored categorical and numerical features using .value_counts() and .describe().
Data Preprocessing

Ignored Cabin due to high missing values.
Handled missing Age values with median imputation using SimpleImputer.
Created a pipeline for numerical features (Age, SibSp, Parch, Fare).
Built a categorical pipeline using OneHotEncoder after filling missing values with the most frequent ones.
Combined numerical and categorical pipelines using FeatureUnion.
Model Training & Evaluation

Trained an SVM classifier (SVC) and Random Forest classifier.
Performed 10-fold cross-validation to compare models.
Achieved ~73% accuracy with SVM and ~81% accuracy with Random Forest.

## Digits

1. Binary Classification with SGD Classifier
Implementing a Stochastic Gradient Descent (SGD) Classifier for binary classification.
Training the model to distinguish between two classes efficiently.
Utilizing decision functions to analyze confidence scores and decision boundaries.
Evaluating model performance using accuracy metrics and confusion matrices.
Handling data scaling with StandardScaler to improve classification results.
2. Multiclass Classification
Using SGDClassifier, RandomForestClassifier, and Naive Bayes for multiclass classification.
Implementing One-vs-All (OvA) and One-vs-One (OvO) strategies for binary-based classifiers.
Generating confusion matrices and normalized confusion matrices to analyze misclassification.
Visualizing misclassified images to detect patterns in model errors.
3. Multilabel Classification
Training a K-Nearest Neighbors (KNN) classifier to predict multiple labels for each instance.
Defining multiple target labels, such as identifying large digits (7, 8, 9) or odd/even classification.
Evaluating multilabel models using F1-score and cross-validation techniques.
4. Multioutput Classification
Implementing a Multioutput classifier where each instance has multiple target variables.
Training models to predict multiple outputs simultaneously, such as denoising images by predicting pixel values.
Using algorithms like RandomForestClassifier to handle correlated target variables.
Evaluating performance using appropriate metrics for multioutput tasks.


## SpamCLassifier

The Logistic Regression classifier predicts whether an email is spam (1) or ham (0) based on the extracted features.

###Data Processing Pipeline
      
Before classification, we process raw emails into numerical features.

 Convert Emails to Word Counts

preprocess_pipeline = Pipeline([

    ("email_to_wordcount", EmailToWordCounterTransformer()),
    
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
    
])

EmailToWordCounterTransformer():

Extracts text from emails.

Removes headers, punctuation, numbers, and converts text to lowercase.

Uses stemming to reduce words to their root form.

Converts the email into a dictionary of word counts.

WordCounterToVectorTransformer():

Creates a sparse matrix of word frequencies.

Each email becomes a feature vector of word counts.

 Example Output:

email 1 → {'buy': 2, 'viagra': 1, 'free': 3, 'win': 2}

email 2 → {'meeting': 1, 'schedule': 2, 'agenda': 1}

Each email is now represented as a vector of word frequencies.


### Training the Logistic Regression Classifier

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)

log_clf.fit(X_train_transformed, y_train)

Takes transformed email data (X_train_transformed) and the corresponding labels (y_train).

Finds patterns in word usage that differentiate spam from ham.

 
### Making Predictions

y_pred = log_clf.predict(X_test_transformed)

Uses the trained model to classify new emails as spam (1) or ham (0).

How Logistic Regression Makes a Decision

Logistic Regression outputs a probability (0 to 1) for spam.

If P(spam) > 0.5, it's classified as spam (1), otherwise ham (0).

 Example Decision:

P(spam | email) = 0.89  → Spam (1)

P(spam | email) = 0.23  → Ham (0)

### Evaluating the Model

precision = precision_score(y_test, y_pred)  # How many predicted spams are correct?

recall = recall_score(y_test, y_pred)        # How many actual spams were detected?

High precision → The model rarely misclassifies ham as spam.

High recall → The model correctly identifies most spam emails.
