# Disaster Response Pipeline Project
Folling a disaster, you'll get millions of tweet message. The goal of this project is to build a ML model to filter and pull out important messages and classfy them into different category, so that different disaster response organizations can really take care of their part.

# Result
The service is provided by a web app, which you can input message and a prediction. Also there are 3 analytical graph that can give you a glimps of the training dataset (bar chart, word cloud, text summary)

# Service Screen Shots
![image](https://github.com/chengweiiii/Disaster-Response-Pipeline-Project/blob/main/screen_shot1.png)
![image](https://github.com/chengweiiii/Disaster-Response-Pipeline-Project/blob/main/screen_shot2.png)
![image](https://github.com/chengweiiii/Disaster-Response-Pipeline-Project/blob/main/screen_shot3.png)

# Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
    - To run text analysis (word cloud, text summarization) 
        `python models/produce_analytic.py`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

# Performance Evaluation
avg precision:0.70, avg recall:0.34, avg f1-score:0.43, test data: 12638
