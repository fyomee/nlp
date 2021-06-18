import sklearn

def MultinomialNB():
    # Here, I add movie_reviews from nltk.corpus manually
    moviedir = r'movie_reviews'

    # loading all files using load_files function because load_files function automatically divides the dataset into data and target sets.
    from sklearn.datasets import load_files
    movie = load_files(moviedir, shuffle=True)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(movie.data, movie.target, train_size = 0.70, test_size = 0.30, random_state = 12)

    # Using all 25K words.
    # CountVectorizer some of the important default parameters are following
    # 1. lowercase bool, default=True; Convert all characters to lowercase before tokenizing.
    # 2. token_patternstr, default=r”(?u)\b\w\w+\b”; The default regexp select tokens of 2 or more alphanumeric characters
    #           (punctuation is completely ignored and always treated as a token separator).
    # 3. ngram_rangetuple (min_n, max_n), default=(1, 1); The lower and upper boundary of the range of n-values for different
    #            word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used.
    #            In here an ngram_range of (1, 1) means only unigrams.
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    movie_count_vectorizer = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))

    # The fit_transform function of the CountVectorizer class converts text documents into corresponding numeric features.
    x_train_counts = movie_count_vectorizer.fit_transform(x_train)

    # Convert raw frequency counts into TF-IDF values
    # The TF stands for "Term Frequency" while IDF stands for "Inverse Document Frequency".
    from sklearn.feature_extraction.text import TfidfTransformer
    movie_tfidf_transformer = TfidfTransformer()

    x_train_tfidf = movie_tfidf_transformer.fit_transform(x_train_counts)

    # Using the fitted vectorizer and transformer, tranform the test data
    x_test_counts = movie_count_vectorizer.transform(x_test)
    x_test_tfidf = movie_tfidf_transformer.transform(x_test_counts)


    # ------------------------------- Multinominal Naive Bayes ------------------------
    from sklearn.naive_bayes import MultinomialNB
    multinominal_nb_classifier = MultinomialNB()
    multinominal_nb_classifier.fit(x_train_tfidf, y_train)

    # Predict the Test set results
    y_pred = multinominal_nb_classifier.predict(x_test_tfidf)

    # --------- print accuracy_score, classification_report, confusion_matrix ----------

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print('Accuracy Score : {0:0.2f}'.format(accuracy_score(y_test, y_pred)))

    print("\nClassification Report : ")
    print(classification_report(y_test,y_pred))

    print("Confusion Matrix : ")
    print(confusion_matrix(y_test,y_pred))

    # ------------- print prediction depending on user input from file ----------------

    # Using a file for input review 'movieReview.in'
    # If want to check for other data just enter the new input in file movieReview.in
    # In here input seperated line by line
    file = open('movieReview.in')
    input_reviews = []
    for sentence in file:
        input_reviews.append(sentence[:-1])

    input_reviews_counts = movie_count_vectorizer.transform(input_reviews)
    input_reviews_tfidf = movie_tfidf_transformer.transform(input_reviews_counts)

    prediction = multinominal_nb_classifier.predict(input_reviews_tfidf)

    print("\n\nPrediction of User inputs: ")
    for review, category in zip(input_reviews, prediction):
        print('%r => %s' % (review, movie.target_names[category]))


def main():
    MultinomialNB()

main()
