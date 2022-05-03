# nlp
# Checking Student Script
 * a fill-in the gap question file *question.in* that contains a text with some gaps,
 * a student script that contains the student answers *script.in* file with the gaps in the text filled, and
 * the *answer.in* file that contains the list of tokens to be placed in the gaps in the order of their appearance in *question.in*.
 Those tokens are stemmed using the Porter's stemmer.

 Write a Python program *scriptChecker.py* that checks the student script. Program should extract from the student script the answers and compare them with the
 list of tokens in the *answer.in* file.

# Average Similarity
 Given 2 text documents, write a function that computes the similarity between the 2 documents using the cosine similarity measure. The documents are represented
 using the TF-IDF. Apply that function to the Movie Review corpus to compute the average similarity of the positive and the negative reviews.

# Classifier
# naiveBayesClassifier.py
 Using the Movie Reviews corpus available in NLTK, write a Python program *naiveBayesClassifier.py* that classify movies according to their sentiment polarity,
 and evaluate your program. Should use the Naive Bayes classifier, and split the data into training and testing (e.g., 70\% and 30\%).

# logisticRegressionClassifier.py
 Using the Movie Reviews corpus available in NLTK, write a program *logisticRegressionClassifier.py* that classify movies according to their sentiment polarity,
 and evaluate your program. Should use the logistic regression as a classifier.

# multilayerNeuralNetworkClassifier.py
 Using the Movie Review corpus, write a program that classify reviews according to their positive/negative polarity, and evaluate the results of the progrh 5 units
 each). Split the data into training and testing (e.g., 90\% and 10\%).am. Should use Word2Vec to represent the words, and the Multi-layer neural network classifier
 (about 10 hidden layers with 5 units each). Split the data into training and testing (e.g., 90\% and 10\%).

# Generating Random Sentence
 Using a language model generated from the Brown corpus which is available in NLTK, write a Python program *shannon.py* that generates random sentences
 (i.e., Shannon Visualization Method) using:
  * bigrams
  * trigrams

# NewsGroups Classification

 Data:
 Newsgroups
 Link: http://qwone.com/~jason/20Newsgroups/
 20news-19997.tar.gz - Original (16MB)
 The data is organized into 20 different newsgroups, each corresponding to a different topic.
 Could be used for News classification.

 Classifier:
 • naive Bayes
 • logistic regression
 • a neural network with few hidden layers with a few units each.
