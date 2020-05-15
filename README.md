# Disaster Response Pipeline Project

 - Skills : 

### Table of Contents

	1. Project Summary
	2. Installation
	3. Instructions
	4. File Description

### 1. Project Summary

This project focus on practice data engineering skills, including building ETL pipeline and ML pipeline using sklearn.   


### 2. Installation
1. sqlachemy
2. ntlk
3. sklearn
4. pickle



### 3. Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
