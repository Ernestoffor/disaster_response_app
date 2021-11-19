# import libraries
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sqlalchemy import create_engine 
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
import time
import re
from sklearn.tree import DecisionTreeClassifier
import pickle
import sys


def load_data(database_filepath):
    """
    A procedure to load dataframe from a sqlite database and return X, Y and category names
    
    INPUT/ARG:
        database_filepath-> database filepath
    OUTPUTS/RETURNS
    . X -> The input variable to the model pipeline
    . Y -> The multi-output parameters of the model
    . category-names: Names of the categories to be predicted
    
        
    """
    engine = create_engine(f'sqlite:///{database_filepath}')

    conn = engine.connect()

    df = pd.read_sql_table('ETLTable', con=conn)
    # Get the input parameter
    X = df.iloc[:,1]

    # Get the multi-output parameters Y
    Y = df.iloc[:, 4:]
    
    # Set the category names to columns of Y
    category_names = Y.columns
    # Convert the Y to numpy (1-d array) array
    Y = Y.values

    return X, Y, category_names



def tokenize(text):
    """
    A Procedure to breakdown a text document into array tokens of words without stopwords
    
    INPUT:
        text: text document
    """
    # Remove special characters
    text = re.sub("[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    stop_words = stopwords.words('english')
    tokens = [w for w in words if w not in stop_words]
    # Initialize a lemmatizer
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = lemmatizer.lemmatize(tok, pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens
    


def build_model():
    """
    A procedure to get a NLP pipeline using sklearn Pipeline, FeatureUnion and MultiOutputClassifier
    With AdaBoostClassifier as the estimator
    
    Return/Output:
        a trained model
    """
    # define a pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('count_vectorizer', CountVectorizer(tokenizer=tokenize)),
                ('tfidf_transformer', TfidfTransformer())
            ]))
            
        ])),

        ('classifier', MultiOutputClassifier(AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=3) , learning_rate = 0.01, n_estimators = 10)))
    ])
    
    # Return the pipeline as a model
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    A procedure to evaluate a trained model or pipeline and compare Y_test with predicted result
    
    INPUTS:
        1. model -> a trained model
        2. X_test -> variable to be used for testing
        3. Y_test -> an array of variables to be predicted
        4. category_names -> Different categories to be predicted 
        
    Returns:
            -> None
    """
    
    
    # predict Y_test from the model
    y_pred = model.predict(X_test)
    
    # Display the metrics showing precision, recall, f1-score, support and accuracy for each category name
    print("The model's metrics for each category are as follows:")
    #for ix, category in enumerate(category_names):
       # print(f"{category}: {'*'*45}")
       # print(classification_report(Y_test[:, ix], y_pred[:, ix]))
       # print('Accuracy in prediction is {0:.0%} \n\n'.format(accuracy_score(Y_test[:, ix], y_pred[:, ix])))

    print(classification_report(Y_test, y_pred, target_names=category_names))

def save_model(model, model_filepath):
    # Export and save the model to the model_filepath a pickle file
    pickle.dump(model, open(model_filepath, "wb"))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()