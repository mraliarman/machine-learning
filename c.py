# phase 1 requirements 
import re
import nltk
import pandas as pd
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# phase 2 requirements 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# phase 3 requirements 
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Step 1: Open output.xlsx
def open_excel_file(file_path):
    return pd.read_excel(file_path)

# Step 2: Preprocess data
def preprocess_data(df):
    # Remove white space, numbers, and non-alphabetic characters from 'reviewText'
    df['reviewText'] = df['reviewText'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x).strip())

    # Remove stop words using NLTK
    # nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))

    # Correct spelling of words using TextBlob
    df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([str(TextBlob(word).correct()) for word in x.split()]))

    # Stemming words using SnowballStemmer
    stemmer = SnowballStemmer('english')
    df['reviewText'] = df['reviewText'].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))

    return df

# Step 3: Save file to line1.xlsx
def save_to_excel(df, output_file):
    df.to_excel(output_file, index=False)

# Step 4: Open line1.xlsx and do Vectorizer words
def open_and_vectorize(file_path):
    df = pd.read_excel(file_path)

    # Handle missing values in 'reviewText' by replacing NaN with an empty string
    df['reviewText'].fillna('', inplace=True)
    
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['reviewText'])
    return X, df['target'], vectorizer, df

# Step 5: Add attributes like n-gram and tf-idf for data
def add_attributes(df, vectorizer, method='tfidf', ngram_range=(1, 1)):
    X = vectorizer.transform(df['reviewText'])
    
    if method == 'tfidf':
        vectorizer.set_params(**{'ngram_range': ngram_range})
        X = vectorizer.transform(df['reviewText'])
    
    return X

# Step 6: Split data
def split_data(X, y, test_size=0.2, random_state=17, shuffle=False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, shuffle=shuffle)
    return X_train, X_test, y_train, y_test

# Step 7: naive bayesian classification
def nb_train(X_train, y_train):
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    return nb_model

# step 8: SVC classification
def train_svc_model(X_train, y_train, kernel='rbf', C=1.0, gamma='scale'):
    """
    Train a Support Vector Classifier (SVC) model.

    Parameters:
    - X_train: The feature matrix for training.
    - y_train: The target variable for training.
    - kernel: Specifies the kernel type ('linear', 'poly', 'rbf', 'sigmoid', etc.).
    - C: Regularization parameter. The strength of the regularization is inversely proportional to C.
    - gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'. If 'scale', it uses 1 / (n_features * X.var()).

    Returns:
    - svc_model: Trained SVC model.
    """
    svc_model = SVC(kernel=kernel, C=C, gamma=gamma)
    svc_model.fit(X_train, y_train)
    return svc_model

# step 9: diction tree
def train_cart_model(X_train, y_train, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1):
    """
    Train a Classification and Regression Trees (CART) model.

    Parameters:
    - X_train: The feature matrix for training.
    - y_train: The target variable for training.
    - criterion: The function to measure the quality of a split ('gini' for Gini impurity, 'entropy' for information gain).
    - max_depth: The maximum depth of the tree. If None, nodes are expanded until all leaves are pure or contain less than min_samples_split samples.
    - min_samples_split: The minimum number of samples required to split an internal node.
    - min_samples_leaf: The minimum number of samples required to be at a leaf node.

    Returns:
    - cart_model: Trained CART model.
    """
    cart_model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    cart_model.fit(X_train, y_train)
    return cart_model

# step 10: logistic regression
def train_logistic_regression_model(X_train, y_train, penalty='l2', C=1.0, max_iter=100, random_state=None):
    """
    Train a Logistic Regression model.

    Parameters:
    - X_train: The feature matrix for training.
    - y_train: The target variable for training.
    - penalty: Used to specify the norm used in the penalization ('l1' or 'l2').
    - C: Inverse of regularization strength; smaller values specify stronger regularization.
    - max_iter: Maximum number of iterations for optimization algorithms.
    - random_state: Seed for random number generation.

    Returns:
    - logistic_regression_model: Trained Logistic Regression model.
    """
    logistic_regression_model = LogisticRegression(penalty=penalty, C=C, max_iter=max_iter, random_state=random_state)
    logistic_regression_model.fit(X_train, y_train)
    return logistic_regression_model

# Step 11 : MLP
def train_mlp_model(X_train, y_train, hidden_layer_sizes=(50,), activation='relu', solver='adam', alpha=0.0001, max_iter=200):
    """
    Train a Multi-layer Perceptron (MLP) model.

    Parameters:
    - X_train: The feature matrix for training.
    - y_train: The target variable for training.
    - hidden_layer_sizes: The number of neurons in each hidden layer.
    - activation: Activation function for the hidden layers ('identity', 'logistic', 'tanh', 'relu').
    - solver: The solver for weight optimization ('lbfgs', 'sgd', 'adam').
    - alpha: L2 regularization term.
    - max_iter: Maximum number of iterations.

    Returns:
    - mlp_model: Trained MLP model.
    """
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, max_iter=max_iter)
    mlp_model.fit(X_train, y_train)
    return mlp_model

# Step 12: xgboost
def train_xgboost_model(X_train, y_train, learning_rate=0.1, n_estimators=100, max_depth=3, subsample=1.0, colsample_bytree=1.0):
    """
    Train an XGBoost model.

    Parameters:
    - X_train: The feature matrix for training.
    - y_train: The target variable for training.
    - learning_rate: Step size shrinkage used in the update to prevent overfitting.
    - n_estimators: Number of boosting rounds (trees) to be run.
    - max_depth: Maximum depth of the decision tree.
    - subsample: Fraction of samples used for fitting the individual base learners.
    - colsample_bytree: Fraction of features used for fitting the individual base learners.

    Returns:
    - xgboost_model: Trained XGBoost model.
    """
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)

    xgboost_model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, subsample=subsample, colsample_bytree=colsample_bytree)
    xgboost_model.fit(X_train, y_train_encoded)
    return xgboost_model

# Step 13: measure the models
def measure(model, X_test, y_test, model_name):
    # Predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot classification report
    class_report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df = class_report_df[class_report_df['support'] > 0]  # Filter out classes with no true samples

    sns.heatmap(class_report_df.iloc[:, :-1].T, annot=True, ax=ax1, cmap='Blues', cbar=False, fmt=".2f")
    ax1.set_title('Classification Report')

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test), ax=ax2)
    ax2.set_title('Confusion Matrix')

    # Show Precision, Recall, F1 Score, and classification report on the figure
    fig.suptitle(f'Model: {model_name}\nPrecision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')

    # Save the figure
    fig.savefig(f'{model_name}_evaluation_figure.png')

    # Show the figure
    plt.show()


# Main execution
if __name__ == "__main__":

    # dont need to uncomment step 1 to 3, the output is already generated in line1.xlsx (as step 1 in project steps)

    # # Step 1: Open output.xlsx
    # excel_file_path = 'output.xlsx'
    # df = open_excel_file(excel_file_path)

    # # Step 2: Preprocess data
    # df_processed = preprocess_data(df)

    # # Step 3: Save file to line1.xlsx
    # output_file_path = 'line1.xlsx'
    # save_to_excel(df_processed, output_file_path)

    # Step 4: Open line1.xlsx and do Vectorizer words
    line1_file_path = 'line1.xlsx'
    X, y, vectorizer, df = open_and_vectorize(line1_file_path)

    # Step 5: Add attributes like n-gram and tf-idf for data
    ngram_range = (1, 1)  # Set your desired n-gram range, e.g., (1, 2) for unigrams and bigrams
    X = add_attributes(df, vectorizer, method='tfidf', ngram_range=ngram_range)

    # Step 6: Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=17, shuffle=False)

    # Step 7: naive bayesian
    nb_model = nb_train(X_train, y_train)
    measure(nb_model, X_test, y_test, 'nb_model')

    # Step 8: svc
    # svc_model = train_svc_model(X_train, y_train)
    # measure(svc_model, X_test, y_test, 'svc_model')

    # Step 9: decision tree
    # cart_model = train_cart_model(X_train, y_train)
    # measure(cart_model, X_test, y_test, 'cart_model')

    # step 10: logistic regression
    # logistic_regression_model = train_logistic_regression_model(X_train, y_train)
    # measure(logistic_regression_model, X_test, y_test, 'logistic_regression_model')

    # Step 11 : MLP
    # mlp_model = train_mlp_model(X_train, y_train)
    # measure(mlp_model, X_test, y_test, 'mlp_model')

    # Step 12: xgboost
    # xgboost_model = train_xgboost_model(X_train, y_train)
    # measure(xgboost_model, X_test, y_test, 'XGBoost')

    # step 13: check and use model
    test_sentence = ["this movie was good"]
    X_test_sentence = vectorizer.transform(test_sentence)
    predictions = nb_model.predict(X_test_sentence)
    print(predictions)