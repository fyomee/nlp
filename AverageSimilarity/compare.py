import math
from numpy import dot
from numpy.linalg import norm
from nltk.corpus import movie_reviews

def cos_similarity(doc1, doc2):
    return dot(doc1, doc2)/(norm(doc1)*norm(doc2))

def main():
    pos_doc = []
    neg_doc = []

    for fileid in movie_reviews.fileids('pos'):
        pos_doc.append(movie_reviews.words(fileid))

    for fileid in movie_reviews.fileids('neg'):
        neg_doc.append(movie_reviews.words(fileid))

    pos_documents = [' '.join(item) for item in pos_doc]
    neg_documents = [' '.join(item) for item in neg_doc]

    # print(len(pos_documents))
    # print(len(neg_documents))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from nltk.corpus import stopwords
    # In here I use the nltk stop_words instead of there ones stop_words='english'
    movie_tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words=stopwords.words('english'))

    pos_reviews_tfidf = movie_tfidf_vectorizer.fit_transform(pos_documents)
    neg_reviews_tfidf = movie_tfidf_vectorizer.fit_transform(neg_documents)

    pos_reviews_array = pos_reviews_tfidf.toarray()
    neg_reviews_array = neg_reviews_tfidf.toarray()

    pos_cos_similarities = 0
    pos_similarity_count = 0
    #In the first loop we don't need to traverse to the last one becasue we don't have any later document which we will comare the last one.
    for i in range(len(pos_reviews_array) - 1):
        for j in range(i+1, len(pos_reviews_array)):
            pos_cos_similarities = pos_cos_similarities + cos_similarity(pos_reviews_array[i], pos_reviews_array[j])
            pos_similarity_count = pos_similarity_count + 1
            # print(pos_cos_similarities)
            # print(i)
            # print(j)
            # print(pos_similarity_count)

    # print("total_pos_cos_similarities")
    # print(pos_cos_similarities)
    # print("pos_similarity_count")
    # print(pos_similarity_count)
    # print("Average Positive Similarity")
    # print(pos_cos_similarities/pos_similarity_count)
    print('Average Positive Similarity : {0:0.2f} %'.format((pos_cos_similarities/pos_similarity_count) * 100))

    neg_cos_similarities = 0
    neg_similarity_count = 0
    #In the first loop we don't need to traverse to the last one becasue we don't have any later document which we will comare the last one.
    for i in range(len(neg_reviews_array) - 1):
        for j in range(i+1, len(neg_reviews_array)):
            neg_cos_similarities = neg_cos_similarities + cos_similarity(neg_reviews_array[i], neg_reviews_array[j])
            neg_similarity_count = neg_similarity_count + 1
            # print(neg_cos_similarities)
            # print(i)
            # print(j)
            # print(neg_similarity_count)

    # print("total_neg_cos_similarities")
    # print(neg_cos_similarities)
    # print("neg_similarity_count")
    # print(neg_similarity_count)
    # print("Average Negative Similarity")
    # print(neg_cos_similarities/neg_similarity_count)

    print('Average Negative Similarity : {0:0.2f} %'.format((neg_cos_similarities/neg_similarity_count) * 100))


main()
