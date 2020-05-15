import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Function to tokenize user input messages

    Input:
        text

    Returns:
        clean tokens
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('cleanedData', engine)

# load model
model = joblib.load("models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """Function to render index page of web app
        with visualizations plotted with plotly

    Input:
        None

    Returns:
        render index page of web app and show two visualizations

    """

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.sum()[3:]
    category_names = list(df.sum()[3:].index)

    category_counts_aidRelated = df[df['aid_related']==1].sum()[3:]

    category_counts_NotAidRelated = df[df['aid_related']==0].sum()[3:]


    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },

	 {
            'data':[
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }

        },

        {
            'data':[
                Bar(
                    x=category_names,
                    y=category_counts_aidRelated
                    )
            ],

            'layout':{
                'title': 'Distribution of Categories for Aid-related Messages',
                'yaxis':{
                    'title':"Count"
                },
                'xaxis':{
                    'title':"Categories"
                }

            },

        },
        {
            'data':[
                Bar(
                    x=category_names,
                    y=category_counts_NotAidRelated
                    )
            ],

            'layout':{
                'title': 'Distribution of Categories for Non-aid-related Messages',
                'yaxis':{
                    'title':"Count"
                },
                'xaxis':{
                    'title':"Categories"
                }

            },

        }

    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """Function to take user input and call saved ML model to classify the message

    Input:
        None

    Returns:
        render the page that shows classification result
    """
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
