import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Function to load raw data

    Input :
        messages_filepath : path to disaster_messages.csv file
        categories_filepath: path to disaster_categories.csv filepath

    Return:
        pandas dataframe that merges two raw data files

    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories,how='left',on=['id'])


def clean_data(df):
    """Function to clean raw data.

    Input:
        dataframe from the load_data() Function

    Returns:
        cleaned dataframe that have converted categories to columns,
        encoded these columns as binary variables, and dropped duplicates

    """


    categories= df['categories'].str.split(';',expand=True)
    row = categories.iloc[0].str.slice(stop=-2)
    category_colnames = list(row)
    categories.columns = category_colnames

    for column in categories:
        categories[column] = categories[column].str.slice(start=-1)
        categories[column] = pd.to_numeric(categories[column])

    indexNames = categories[categories['related'] == 2 ].index
    categories.drop(indexNames,inplace=True)

    df=df.drop('categories',axis=1)
    df = df.drop(columns=['original'])
    df = pd.concat([df,categories],axis=1,join='inner')
    df=df.drop_duplicates(['message'])
    return df


def save_data(df, database_filename):
    """Function to save the cleaned dataframe to sqlite database

    Input:
        cleaned dataframe
        datasets_filename (including filepath)

    Return:
        None

    """

    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('cleanedData', engine,if_exists='replace',index=False)


def main():
    """ Function to call the previously defined function and execute
        the ETL pipeline.
    """

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
