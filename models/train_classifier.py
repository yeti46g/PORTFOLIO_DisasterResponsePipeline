import sys
from sqlalchemy import create_engine
import pandas as pd
import re
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle

def load_data(database_filepath):
    """ Function to load data from SQLite database

    Input:
        database_filepath

    Returns:
         X (str): pandas series contains message texts.
         Y : dataframe contains 36 category labels and their binary values
         a list of the 36 label names
    """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('cleanedData',engine)
    X = df['message']
    Y = df[df.columns[3:39]]
    return X,Y, list(df.columns[3:39])


def tokenize(text):
    """ Function to tokenize messages

    Input:
        text : str

    Returns:
        tokens

    """
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower().strip())
    tokens = word_tokenize(text)
    tokens = [WordNetLemmatizer().lemmatize(word) for word in tokens if word not in stopwords.words("english")]
    return tokens


def build_model():
    """ Function to build machine learning pipeline
    Input:
        None

    Returns:
        Pretrained model

    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])

    parameters = {
        # since searching through many hyper parameters on my local machine takes
        # a significantly long time, here I reduced the search space to only two different numbers
        # of neighbors in the KNNclassifier

        #'vect__ngram_range': ((1, 1), (1, 2)),
        #'vect__max_df': (0.5, 0.75, 1.0),
        #'vect__max_features': (None, 5000, 10000),
        #'tfidf__use_idf': (True, False),
        'clf__estimator__n_neighbors':[3,5]
        #'clf__estimator__weights':['uniform','distance'],
        #'clf__estimator__metric':['euclidean','manhattan']
    }

    cv =  GridSearchCV(pipeline, param_grid=parameters)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """Function to evaluate trained model

    Input:
        model: trained model
        X_test: message texts of test data
        Y_test: labels of test data
        category_names: list of the 36 category target_names

    Returns:
        print precision, recall, f1-score and support for each label

    """
    y_pred = model.predict(X_test)
    print(classification_report(Y_test, y_pred,target_names=category_names))


def save_model(model, model_filepath):
    """Function to save trained model to pickle file

    Input:
        trained model

    Returns:
        model in pickle file

    """
	with open(model_filepath,'wb') as f:
		pickle.dump(model,f)

def main():
    """ Function to call the previous functions and execute ML pipeline
    """


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
