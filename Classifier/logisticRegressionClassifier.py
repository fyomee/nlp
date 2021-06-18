import sklearn
# import pandas as pd
import numpy as np

def LogisticRegressionClassifier():
    # Here, I add movie_reviews from nltk.corpus manually
    moviedir = r'movie_reviews'

    # loading all files using load_files function because load_files function automatically divides the dataset into data and target sets.
    from sklearn.datasets import load_files
    movie = load_files(moviedir, shuffle=True)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(movie.data, movie.target, train_size = 0.70, test_size = 0.30, random_state = 12)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    movie_tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words=stopwords.words('english'))

    x_train_tfidf = movie_tfidf_vectorizer.fit_transform(x_train)
    x_test_tfidf = movie_tfidf_vectorizer.transform(x_test)


    # ------------------------------- Logistic Regression Classifier------------------------
    from sklearn.linear_model import LogisticRegression
    regressor = LogisticRegression()
    regressor.fit(x_train_tfidf, y_train)

    # print(regressor.intercept_)
    # print(regressor.coef_)

    # Predict the Test set results
    y_pred = regressor.predict(x_test_tfidf)

    # df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
    # print(df)

    # --------- print mean_absolute_error, mean_squared_error, root_mean_squared_error ----------

    from sklearn import metrics
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # --------- print accuracy_score, classification_report, confusion_matrix ----------
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print('Accuracy Score : {0:0.2f}'.format(accuracy_score(y_test, y_pred)))

    print("\nClassification Report : ")
    print(classification_report(y_test,y_pred))

    print("Confusion Matrix : ")
    print(confusion_matrix(y_test,y_pred))

    # # Use score method to get accuracy of model
    # score = regressor.score(x_test_tfidf, y_test)
    # print(score)
    # cm = metrics.confusion_matrix(y_test, y_pred)
    # print(cm)

    # ------------- print prediction depending on user input from file ----------------

    # Using a file for input review 'movieReview.in'
    # If want to check for other data just enter the new input in file movieReview.in
    # In here input seperated line by line
    file = open('movieReview.in')
    input_reviews = []
    for sentence in file:
        input_reviews.append(sentence[:-1])

    input_reviews_tfidf = movie_tfidf_vectorizer.transform(input_reviews)

    prediction = regressor.predict(input_reviews_tfidf)

    print("\n\nPrediction of User inputs: ")
    for review, category in zip(input_reviews, prediction):
        print('%r => %s' % (review, movie.target_names[category]))


def main():
    LogisticRegressionClassifier()

main()
