# Fake-News-Detection-using-Spark

The objective of this project is to implement a model to detect fake news using Apache Spark in Pyspark in Google colab. 

## Libraries Installed: 

Three libraries are installed using the below command to implement this project. Pyspark is installed to code the project in Apache Spark using python. Elephas is installed to integrate keras with Spark. Elephas supports certain versions of keras and tensforflow. Keras version 2.2.4 and TF version 1.14.0 is installed. 

pip install pyspark

pip install q keras==2.2.4

pip install q tensorflow==1.14.0

pip install elephas

## Dataset

The fake news dataset is taken from kaggle: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset. There are two seperate csv files for real and fake news. The two data are combined to a single dataset by creating labels: Real and Fake. Each dataset has title, text, subject and date features.The fake news detection is performed using three different features: title, text of the news article and by concatanating title and text of the news.

## Modelling

The text processing is performed using RegexTokenizer, Word2Vec and StringIndexer. The fake news detection is implemented by training the data using three algorithms: Decision Tree, Gradient Boosting and Neural Network. The parmeter tuning for Gradient Boosting is done using 3-fold Cross Validation. The neural network is implemented by integrating Keras with Elephas to run the network on Apache Spark. The model evaluation is performed by studying the accuracy, AUC and F1 score and confusion matrix.

## Results

Among the three models, Neural network performed the best for all the three features: title, text and title-text. The below are the results for the three models implemented.

|Feature   |Model                                               |Accuracy                  |No. of False positive|No. of False neagative|
|----------|----------------------------------------------------|--------------------------|---------------------|----------------------|
|Title     |Decision Tree<br>Gradient Boosting<br>Neural Network|88.07%<br>90.06%<br>91.41%|477<br>430<br>405    |601<br>468<br>371     |
|Text      |Decision Tree<br>Gradient Boosting<br>Neural Network|90.08%<br>93.83%<br>98.53%|571<br>256<br>48     |313<br>294<br>83      |
|Title-text|Decision Tree<br>Gradient Boosting<br>Neural Network|91%<br>93.79%<br>98.60%   |358<br>262<br>61     |457<br>300<br>66      |

Comparing the results to see the best performance of each model against the individual features, Decision tree performed better with title-text feature, Gradient Boosting with text feature and Neural network with both text and text-title features.

## Link to the Colab code:

Fake_news_detection_using_text: https://colab.research.google.com/drive/1JfAAZ70sWDHcp0QK6DQAvhsCG7OrQrAi?usp=sharing
Fake_news_detection_using_title: https://colab.research.google.com/drive/1LZ5f91ES7hZ6U2NI7GcdURobZtAEyLfD?usp=sharing
Fake_news_detection_using_title_and_text: https://colab.research.google.com/drive/1iOZmYeZi-xHMrtil9IKwBHze2_5zmCPN?usp=sharing

## Reference:

https://spark.apache.org/docs/latest/ml-guide.html

https://github.com/maxpumperla/elephas
