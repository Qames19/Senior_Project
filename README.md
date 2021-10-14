# Project Abstract  

> The goal of this project is to determine if there is any prediction can be made between tweets concerning the following companies:  Apple, Amazon, Google, Microsoft, and Tesla.  The analysis was performed by James Mick who is an undergraduate student at Texas A&M University-San Antonio.  The analysis was performed on a set of tweets which contained the company’s ticker symbols collected over the period of January 1, 2015 through December 31, 2019.  Previous analysis on this dataset has been performed by several researchers and can be found on the dataset’s Kaggle page.  Python was used to consolidate and modify the data before using classifiers to determine if the stock prices were predicted by several different meta-features.  The analysis shows that it is likely that interaction level and sentiment likely have some predictive qualities based on the meta-features added.  The price showed some predictive qualities, but new meta-features are required to find better predictive values.

## File contents

### Senior_Project_James_Mick.py  

This code takes the data provided in the Kaggle page, and preprocesses it into two 5000 tweet samples of 100 and 500 attributes which can be used for further analysis. 

### n-grams, wordclouds, and TFIDF's

These are n-grams and wordclouds that are output by the program, and TFIDF's that were used as inputs for analysis.

### classifiers mentioned in my analysis  

These are files that I used to complete my analysis

### classifiers not mentioned in my analysis  

Feel free to check these out (there are a lot of them).  They have not been optimized in any way.  These are the trees as they appear running each of the data sets

### original data

This contains the data as it was when I downloaded it from kaggle

### outputs from project processor

This includes each of the train_data sets used to create the trees as well as the tweet file as it has been modified by the program project_processor.py

### swearWords.txt

List of swear words used to determine if tweets contained explicit language.

## Project code

### project_processing.py  

This program reads Company_Tweet.csv and Tweet.csv and then modifies them to create interaction_level, price_change, and sentiment.  It also adds meta-features and the bag of words.  Each of the combined training datas are written to an output folder to be used later.  

This program takes ~4 hours to run.  As such, I have included all of the outputs in a separate folder called "outputs_from_project_processor".

### project_classifier.py

This program performs decision tree classifier and regressor as well as random forest classifier.  The program takes the combined TFDIF files from project_processing.py and produces the trees for the aforementioned classifiers.  There are several statements near the top of the file that need to be either commented or uncommented to work with each file.  Additionally, there are variables that need to be changed to ensure the trees are built correctly.

### project_pruner.py

This program takes the TFDIF files and attempts to find the best ccp_alpha for them decision trees and then tries to optimize the selections for random forests.  After running this program, project_classifier can be altered to output the optimized tree results for each training set.  Running all sets is a very time intensive operation; therefore, I only performed this on 3 separate training datas.  

## Project_Report_James_Mick.docx  

This is the final analysis that was submitted upon the conclusion of the senior seminar.