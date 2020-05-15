# Disaster Response Pipeline Project

 - Skills : ETL (sqlachemy, SQLite database), NLP (nltk), ML (sklearn, multi-output classification, imbalanced data ), visualization (plotly), web app (flask, html, css, bootstrap)

### Table of Contents

	1. Project Summary
	2. Installation
	3. Instructions
	4. File Description

### 1. Project Summary

This project focus on practice data engineering skills, including building ETL (Extract, Transform, Load) pipeline and ML (Machine Learning) pipeline using libraries such as sqlalchemy, nltk, sklearn. It also utilizes Plotly to build interactive web-based data visualizations. Additionally, a web app is built to use ML model to classify user inputs using boostrap and Flask. 

These above mentioned skills are applied to analyze disaster data provided by [Figure Eight](https://appen.com/). This data set contains real messages that were sent during disaster events. The machine learning pipeline is created to categorize these messages so that they can me sent to an appropriate disaster relief agency. The web app is built so that an emergency worker can input a new message and get classification results in several categories (multi-output classification).

Please note that this project's emphasis is on software engineering skills (create basic data pipeline and building web app), rather than machine learning modelling. The ML model created here is a baseline model without in-depth tunning. For example, since running the grid search on a lot of hyper-parameters on the author's local machine takes a too long, most hyper-parameters in the code are muted. Therefore, the resulting model is not intented for high accuracy classification. 

### 2. Installation

Except for the regular data science libraries such as numpy, pandas, and matplotlib, the following libraries need to be installed to run the codes in this repo.

1. sqlachemy
2. ntlk
3. sklearn
4. pickle



### 3. Instructions
Please download or clone the entire repo to a root directory, and then:

1. Run the following commands in the **project's root directory** to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database <br>
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves <br>
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the **project's root directory** to run your web app.<br>
    `python app/run.py`

3. Go to http://0.0.0.0:3001/
