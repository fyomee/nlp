import sklearn
import numpy as np

from nltk.tokenize import word_tokenize
from gensim.models import word2vec

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.stem.porter import PorterStemmer
porter = PorterStemmer()

def getProcessedData(texts):
    docs=[]
    for text in texts:
        text = text.lower()
        doc = word_tokenize(text)
        doc = [porter.stem(word) for word in doc if word not in stop_words and word.isalpha()]
        docs.append(doc)
    return docs

def getFeatureVec(words, model, num_features):
    featureVec = np.zeros((num_features,), dtype="float32")
    no_of_words = 0
    index2word_set = set(model.wv.index_to_key)
    for word in words:
        if word in index2word_set:
            nof_of_words = no_of_words + 1
            featureVec = np.add(featureVec, model.wv[word])
    if no_of_words == 0:
        no_of_words = 1
    featureVec = np.divide(featureVec, no_of_words)
    return featureVec

def getAvgFeatureVecs(reviews, model, num_features):
    reviewFeatureVecs = np.zeros((len(reviews), num_features), dtype="float32")
    counter = 0
    for review in reviews:
        reviewFeatureVecs[counter] = getFeatureVec(review, model, num_features)
        counter = counter + 1
    return reviewFeatureVecs

def MLPClassifier():
    from nltk.corpus import movie_reviews

    X = [movie_reviews.raw(fileid) for fileid in movie_reviews.fileids()]
    y = [movie_reviews.categories(fileid)[0] for fileid in movie_reviews.fileids()]

    processed_data = getProcessedData(X)
    # len(processed_data)

    # Split data into training and test sets
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(processed_data, y, train_size = 0.90, test_size = 0.10, random_state = 12)

    vector_size = 300
    model = word2vec.Word2Vec(processed_data, workers=4, vector_size=vector_size, min_count = 1, window = 10, sample = 1e-3)

    f_matrix_train = getAvgFeatureVecs(x_train, model, vector_size)
    # print(len(f_matrix_train))
    f_matrix_test = getAvgFeatureVecs(x_test, model, vector_size)
    # print(len(f_matrix_test))

    from sklearn.neural_network import MLPClassifier
    # clf = MLPClassifier(hidden_layer_sizes=(5, 5, 5), max_iter=500, alpha=0.0001, solver='sgd', random_state=21,tol=0.000000001)
    clf = MLPClassifier(hidden_layer_sizes=(5, 5, 5, 5, 5, 5, 5, 5, 5, 5), max_iter=500, alpha=0.0001, solver='sgd', random_state=21, tol=0.000000001)
    clf.fit(f_matrix_train, y_train)

    y_pred = clf.predict(f_matrix_test)

    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

    # precision tp / (tp + fp)
    precision = precision_score(y_test,y_pred, pos_label="pos")
    print('Precision: %f' % precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test,y_pred, pos_label="pos")
    print('Recall: %f' % recall)
    # # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test,y_pred, pos_label="pos")
    print('F1 score: %f' % f1)

    # accuracy: (tp + tn) / (p + n)
    print('Accuracy Score : {0:0.2f}'.format(accuracy_score(y_test, y_pred)))

    print("\nClassification Report : ")
    print(classification_report(y_test,y_pred,zero_division=0))

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

    processed_input_reviews = getProcessedData(input_reviews)

    input_reviews_matrix = getAvgFeatureVecs(processed_input_reviews, model, vector_size)

    prediction = clf.predict(input_reviews_matrix)

    print("\n\nPrediction of User inputs: ")
    for review, category in zip(input_reviews, prediction):
        print('%r => %s' % (review, category))


def main():
    MLPClassifier()

main()
