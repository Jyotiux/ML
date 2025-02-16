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
