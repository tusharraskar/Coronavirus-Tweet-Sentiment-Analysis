# Sentiment Analysis: Predicting Sentiment of COVID-19 Tweets

## 1. Introduction

- The ongoing COVID-19 pandemic has drastically transformed the way humans communicate. Digital platforms, especially microblogging sites like Twitter, have emerged as essential tools for disseminating information.

- Given the importance of these platforms, a multitude of opinions, perspectives, and sentiments about the pandemic are continuously shared. The sentiments conveyed through these tweets can provide crucial insights into public opinion and mood.

- To analyze these sentiments, we have leveraged a variety of machine learning and deep learning techniques in this project. We aimed to predict sentiment from these COVID-19 related tweets, thus providing a quantitative measure of public opinion.

## 2. Exploratory Data Analysis (EDA)

- Started the analysis with exploratory data analysis which helped to understand the dataset better.

- Created a sentiment distribution plot that indicated a balanced dataset, an essential aspect for the machine learning models.

- Generated a bar plot for the top ten tweeting locations to get an overview of the most active regions regarding COVID-19 discussions.

- Distinguished the locations based on the number of negative and positive tweets to understand the geographical variance of sentiments.

- Evaluated the average number of words and characters in tweets by sentiment to understand the tweet structure.

- By identifying the top mentions and hashtags, I was able to gain insight into the most discussed topics and prevailing trends about the pandemic.

## 3. Data Preprocessing

- The preprocessing stage was a critical part of this project, where raw data was prepared for further analysis and modeling.

- I first performed text cleaning tasks, such as removing unwanted characters, tokenizing the text, and lowercasing, to normalize the text data for the subsequent steps.

- Stop words were then removed, as they do not contribute significantly to the sentiments expressed. This helped reduce the noise in the dataset.

- To deal with the problem of different-length tweets, I used the padding technique, which ensured that all sequences had the same length, thereby facilitating their processing by the machine learning models.

- The en_core_web_lg model was loaded. This model is part of the Spacy library, which is used for advanced Natural Language Processing tasks.

- This model helped me convert the text information into numerical vectors that the machine learning and deep learning algorithms can understand and process.

- I created an embedding matrix using this model. The embedding matrix essentially provided a dense representation of words, preserving their semantic relationships.

- These preprocessed features were then fed into the various models for training. By ensuring that the data fed into the models was as clean and structured as possible, I increased the likelihood of achieving accurate predictions.

## 4. Model Training 

Various machine learning models were trained:

- Logistic Regression
- Stochastic Gradient Descent (SGD)
- Decision Trees
- Random Forests
- AdaBoost
- Naive Bayes
- MultinomialNB
- Artificial Neural Network (ANN)
- Recurrent Neural Network (RNN)
- Long short-term memory (LSTM)

## 5. Model Evaluation and Performance

- I used a variety of machine learning and deep learning models, each with its strengths and limitations. Here's a brief overview of the performance.

- The Logistic Regression models demonstrated impressive performance, achieving an accuracy of 83.6% on the test set after hyperparameter tuning.

- The SGD models also performed well, achieving an accuracy of 83.6% on the test set after hyperparameter tuning.

- Even though the Decision Trees and Random Forest models showed almost perfect performance on the training data, they performed lower on the test set, indicating an overfitting problem.

- The AdaBoost models presented balanced performance, achieving about 65% accuracy on both the training and test sets.

- Among the deep learning models, the LSTM model trained with the Adam optimizer stood out, achieving an accuracy of 86.8% on the test set.

## 6. Conclusion

- Considering the performance of the models, it's clear that machine learning and deep learning techniques hold significant potential for sentiment analysis tasks, especially in the context of global events like a pandemic.

- The accuracy results from models like Logistic Regression, SGD, and LSTM demonstrate the applicability of these techniques for analyzing and predicting public sentiment on a wide range of topics.

- In the evolving digital age, the ability to comprehend and predict public sentiment becomes more critical across various sectors. Such data-driven insights can help decision-makers respond more effectively to public sentiment, potentially leading to more positive outcomes in tackling issues like the COVID-19 pandemic.
