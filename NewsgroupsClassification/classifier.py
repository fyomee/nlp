from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def getData():
    # The whole dataset is organized into 20 newsgroups
    # Each record in the dataset is actually a text file (a series of words)
    newsdir = r'20_newsgroups'

    # loading all files using load_files function because load_files function automatically divides the dataset into data and target sets.
    from sklearn.datasets import load_files
    news = load_files(newsdir, shuffle=True, encoding='ISO-8859-1')

    # After loading file news object has properties like data, target and target_names
    print('Data Count : ', len(news.data))
    print('Target Count : ', len(news.target))
    print('Distinct Target Count : ', len(set(news.target)))
    print('Target Names / Newsgroups : ')
    print(news.target_names)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(news.data, news.target, train_size = 0.90, test_size = 0.10, random_state = 12)

    # Since the dataset is a series of words, we need to convert text representations into numerical representations before
    # running any machine learning algorithms.

    # CountVectorizer some of the important default parameters are following
    # 1. lowercase bool, default=True; Convert all characters to lowercase before tokenizing.
    # 2. token_patternstr, default=r”(?u)\b\w\w+\b”; The default regexp select tokens of 2 or more alphanumeric characters
    #           (punctuation is completely ignored and always treated as a token separator).
    # 3. ngram_rangetuple (min_n, max_n), default=(1, 1); The lower and upper boundary of the range of n-values for different
    #            word n-grams or char n-grams to be extracted. All values of n such such that min_n <= n <= max_n will be used.
    #            In here an ngram_range of (1, 1) means only unigrams.
    from sklearn.feature_extraction.text import CountVectorizer
    from nltk.corpus import stopwords
    news_count_vectorizer = CountVectorizer(min_df=1, stop_words=stopwords.words('english'))

    # The fit_transform function of the CountVectorizer class converts text documents into corresponding numeric features.
    x_train_counts = news_count_vectorizer.fit_transform(x_train)

    # Convert raw frequency counts into TF-IDF values
    # The TF stands for "Term Frequency" while IDF stands for "Inverse Document Frequency".
    from sklearn.feature_extraction.text import TfidfTransformer
    news_tfidf_transformer = TfidfTransformer()

    x_train_tfidf = news_tfidf_transformer.fit_transform(x_train_counts)

    # Using the fitted vectorizer and transformer, tranform the test data
    x_test_counts = news_count_vectorizer.transform(x_test)
    x_test_tfidf = news_tfidf_transformer.transform(x_test_counts)

    return x_train_tfidf, y_train, x_test_tfidf, y_test


# ------------------------------- Multinominal Naive Bayes ------------------------
def MultinominalNaiveBayesClassifier(x_train_tfidf, y_train, x_test_tfidf, y_test):

    # Naive Bayes classifier for multinomial models.
    # The multinomial Naive Bayes classifier is suitable for classification with discrete features (e.g., word counts for text classification).
    from sklearn.naive_bayes import MultinomialNB

    # alpha: float, default=1.0
    #        Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).
    # fit_prior: bool, default=True
    #            Whether to learn class prior probabilities or not. If false, a uniform prior will be used.
    multinominal_nb_classifier = MultinomialNB()
    multinominal_nb_classifier.fit(x_train_tfidf, y_train)

    # Predict the Test set results
    y_pred = multinominal_nb_classifier.predict(x_test_tfidf)

    # --------- print accuracy_score, classification_report, confusion_matrix ----------
    print('Accuracy Score : {0:0.2f}'.format(accuracy_score(y_test, y_pred)))

    print('\nClassification Report : ')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix : ')
    print(confusion_matrix(y_test,y_pred))

    # ------------------------------- Logistic Regression Classifier------------------------
def LogisticRegressionClassifier(x_train_tfidf, y_train, x_test_tfidf, y_test):

    # Logistic Regression classifier.
    #
    # In the multiclass case, the training algorithm uses the one-vs-rest (OvR) scheme if the ‘multi_class’ option is set to ‘ovr’,
    # and uses the cross-entropy loss if the ‘multi_class’ option is set to ‘multinomial’.
    from sklearn.linear_model import LogisticRegression

    # solver: {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}, default=’lbfgs’
    #         Algorithm to use in the optimization problem. Default is ‘lbfgs’. To choose a solver, you might want to consider the following aspects:
    #         For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
    #         For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
    #         ‘liblinear’ is limited to one-versus-rest schemes.
    # max_iter: int, default=100
    #           Maximum number of iterations taken for the solvers to converge.
    # multi_class: {‘auto’, ‘ovr’, ‘multinomial’}, default=’auto’
    #              If the option chosen is ‘ovr’, then a binary problem is fit for each label. For ‘multinomial’ the loss minimised is the multinomial
    #              loss fit across the entire probability distribution, even when the data is binary. ‘multinomial’ is unavailable when solver=’liblinear’.
    #              ‘auto’ selects ‘ovr’ if the data is binary, or if solver=’liblinear’, and otherwise selects ‘multinomial’.
    regressor = LogisticRegression(max_iter = 500)
    regressor.fit(x_train_tfidf, y_train)

    # Predict the Test set results
    y_pred = regressor.predict(x_test_tfidf)

    # # --------- print mean_absolute_error, mean_squared_error, root_mean_squared_error ----------
    #
    # from sklearn import metrics
    # print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
    # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    # --------- print accuracy_score, classification_report, confusion_matrix ----------
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    print('Accuracy Score : {0:0.2f}'.format(accuracy_score(y_test, y_pred)))

    print('\nClassification Report : ')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix : ')
    print(confusion_matrix(y_test,y_pred))

    # ------------------------------- Neural Network Classifier------------------------
def NeuralNetworkClassifier(x_train_tfidf, y_train, x_test_tfidf, y_test):
    from sklearn.neural_network import MLPClassifier

    # Multi - layer Perceptron classifier
    # hidden_layer_sizes: tuple, length = n_layers - 2, default=(100,)
    #                     The ith element represents the number of neurons in the ith hidden layer.
    # max_iter: int, default=200
    #           Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations.
    #           For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point
    #           will be used), not the number of gradient steps.
    # solver: {‘lbfgs’, ‘sgd’, ‘adam’}, default=’adam’
    #         The solver for weight optimization.
    #         ‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
    #         ‘sgd’ refers to stochastic gradient descent.
    #         ‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
    #         Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more)
    #         in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.
    # random_state: int, RandomState instance, default=None
    #               Determines random number generation for weights and bias initialization, train-test split if early stopping is used,
    #               and batch sampling when solver=’sgd’ or ‘adam’. Pass an int for reproducible results across multiple function calls.
    # tol: float, default=1e-4
    #      Tolerance for the optimization. When the loss or score is not improving by at least tol for n_iter_no_change consecutive
    #      iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.


    # The following one gives us a better result; however, it took a long time. Since accuracy is not a priority for this assignment,
    # I am commenting on that for a testing purpose.

    # clf = MLPClassifier(hidden_layer_sizes=(100,100,100), max_iter=300, random_state=21, tol=0.000000001) # Accuracy Score : 0.91

    clf = MLPClassifier(hidden_layer_sizes=(10,10,10), max_iter=300, random_state=21, tol=0.000000001) # Accuracy Score : 0.84
    
    clf.fit(x_train_tfidf, y_train)

    y_pred = clf.predict(x_test_tfidf)

    # accuracy: (tp + tn) / (p + n)
    print('Accuracy Score : {0:0.2f}'.format(accuracy_score(y_test, y_pred)))

    print('\nClassification Report : ')
    print(classification_report(y_test, y_pred))

    print('Confusion Matrix : ')
    print(confusion_matrix(y_test,y_pred))

def main():
    x_train_tfidf, y_train, x_test_tfidf, y_test = getData()

    print('\n\n------------------------------- Multinominal Naive Bayes ------------------------\n\n')
    MultinominalNaiveBayesClassifier(x_train_tfidf, y_train, x_test_tfidf, y_test)

    print('\n\n------------------------------- Logistic Regression Classifier ------------------------\n\n')
    LogisticRegressionClassifier(x_train_tfidf, y_train, x_test_tfidf, y_test)

    print('\n\n------------------------------- Neural Network Classifier ------------------------\n\n')
    NeuralNetworkClassifier(x_train_tfidf, y_train, x_test_tfidf, y_test)


main()
