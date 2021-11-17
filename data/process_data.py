import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine 


def load_data(messages_filepath, categories_filepath):
    """
    function to load data from two files as pandas dataframes and merge them
    INPUTS:
        - messages_filepath -> full path to the messages file
        - categories_filepath -> full path to the categories file
    OUTPUT/RETURNS:
        - merged dataframe
    """
    # load the messages dataframe
    messages =  pd.read_csv(messages_filepath)
    
    # load the categories dataframe
    categories = pd.read_csv(categories_filepath)
    
    # merge the two dataframes on 'id' column
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    Function to clean the returned dataframe from the load_data() function
    INPUT:
        df: dataframe returned from the load_data() function
    OUTPUT/RETURNS:
         cleaned dataframe
    """
    # split the categories column on ";" in df into different categories to create new columns
    # put the new columns in a new dataframe called categories
    categories = df['categories'].str.split(pat=';', expand=True)
    
    # select the first row of the categories dataframe and remove the trailing digits and '-'
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[: -2])
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[len(x)-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis = 1, inplace=True)
    # set the column names
    categories.columns = category_colnames
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # replace df['related'] = 2 with 1
    df['related'].replace(2, 1, inplace=True)
    
    return df

def save_data(df, database_filename):
    """
    A procedure to save the cleaned dataframe returned from the clean_data() function above in Sqlite database
    INPUTS:
        - df: -> returned dataframe
        - database_filename: -> The name of the table to save the df in the database
    RETURNS:
        NONE
        
    """
    # create an sqlite database engine using sqlalchemy
    engine = create_engine(f'sqlite:///{database_filename}')
    # save the df in a table
    df.to_sql('ETLTable', engine, index=False, if_exists='replace')
    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()