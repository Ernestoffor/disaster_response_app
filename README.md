# Disaster Response with Natural Language Processing

## Project Overview
In this project, data engineering, NLP and ML skills are used to analyze disaster data from [Figure Eight](https://figure-eight.com) to build a model for an API that classifies disaster messages in real time. The datasets are real  messages sent during disaster events and were curated by `Figure Eight` for the purposes of categorizing these messages into different areas of need such as water, food, etc. The goal of this project is to create a machine learning pipeline to categorize these events so that  messages can be sent to an appropriate disaster relief agency.


## Project Components

There are a number of components of this project, namely: 
1. ETL Preparation Pipeline

The jupyter notebook, `ETL Pipeline Preparation.ipynb`, is used to prepare and develop ETL Pipeline below.

2. ML Preparation Pipeline

The jupyter notebook, `ML Pipeline Preparation.ipynb`, is used to prepare and develop the MLPipeline below.

3. ETL Pipeline

In a Python script inside data folder, `process_data.py` is a data cleaning pipeline that:

    . Loads the messages and categories datasets
    . Merges the two datasets
    . Cleans the data
    . Stores it in a SQLite database

4. ML Pipeline

In a Python script inside models directory, `train_classifier.py`, is a machine learning pipeline that:

    * Loads data from the SQLite database
    * Splits the dataset into training and test sets
    * Builds a text processing and machine learning pipeline
    * Trains and tunes a model using GridSearchCV
    * Outputs results on the test set
    * Exports the final model as a pickle file

5. Flask Web App

The web app includes flask, html, css and javascript for the front end of the application. It is inside `app` directory containing templates with html, css and javascript as well as a run.py file that implements flask for visualizations. 


### Instructions on Running the App:
1. Run the following commands in the project's root directory to set up your database and model.

    *  To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponseETL.db`
    *  To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponseETL.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


