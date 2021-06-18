import nltk
import random
from nltk.corpus import brown

def main():

    # ------------------------ Data Processing: Start -----------------------

    sentence_starting = "<s>"
    sentence_ending = "</s>"
    brown_sents_list = []

    for sentence_token in brown.sents():
        sentence_token_with_start_end_point = [sentence_starting] + sentence_token + [sentence_ending]
        brown_sents_list.extend(sentence_token_with_start_end_point)

    # print(len(brown_sents_list))

    # ----------------------- Data Processing: End --------------------------

    # --------------------------- bigrams: Start ----------------------------
    bigrams = list(nltk.bigrams(brown_sents_list))
    bigrams_cfd = nltk.ConditionalFreqDist(
        (first,second)
        for (first, second) in bigrams[:-1])

    bigrams_cpd = nltk.ConditionalProbDist(bigrams_cfd, nltk.MLEProbDist)

    # Both of the following list are same
    # print(bigrams_cfd[sentence_starting].keys())
    # print(len(bigrams_cfd[sentence_starting].keys()))
    # print(bigrams_cpd[sentence_starting].samples())
    # print(len(bigrams_cpd[sentence_starting].samples()))

    bigram_random_sentence = ""
    given_word = sentence_starting;
    while (given_word != sentence_ending):
        bigram_random_sentence += given_word + " "

        # we can use the ConditionalFreqDist for generating random sentence
        # if given_word in bigrams_cfd:
        #     given_word = random.choice(list(bigrams_cfd[given_word].keys()))

        # we also can use the ConditionalProbDist for generating random sentence
        if given_word in bigrams_cpd:
            # generate() -- generate random word(didnot consider minimum probability threshold) from bigrams_cpd[sentence_starting].samples()
            temp = given_word
            given_word = bigrams_cpd[given_word].generate()
            # print("Probability check for bigrams")
            # print(temp + ' ' + given_word)
            # print(bigrams_cpd[temp].prob(given_word))
        else:
            break
    bigram_random_sentence += sentence_ending
    print("bi-grams random sentence : ")
    print(bigram_random_sentence)

    # ---------------------------- bigrams: End -----------------------------

    # -------------------------- trigrams: Start ----------------------------

    trigrams = list(nltk.trigrams(brown_sents_list))
    # print(len(trigrams))

    trigrams_cfd = nltk.ConditionalFreqDist(
        ((first,second),third)
        for (first, second, third) in trigrams[:-2])

    # print(len(trigrams_cfd.conditions()))

    trigrams_cpd = nltk.ConditionalProbDist(trigrams_cfd, nltk.MLEProbDist)

    previous_word = sentence_starting

    # we can choose random second word with starting word '<s>' by reusing bigrams_cfd
    # current_word = random.choice(list(bigrams_cfd[sentence_starting].keys()))

    # we can choose random second word(didnot consider minimum probability threshold) with starting word '<s>' by reusing bigrams_cpd
    current_word = bigrams_cpd[sentence_starting].generate()

    trigram_random_sentence = previous_word + " "
    while (current_word != sentence_ending):
        trigram_random_sentence += current_word + " "
        index = (previous_word, current_word)

        # we can use the ConditionalFreqDist for generating random sentence
        # if index in trigrams_cfd:
        #     previous_word = current_word
        #     current_word = random.choice(list(trigrams_cfd[index].keys()))

        # we can also use the ConditionalProbDist for generating random sentence
        if index in trigrams_cpd:
            previous_word = current_word
            # generate() -- generate random word(didnot consider minimum probability threshold) from trigrams_cpd[sentence_starting].samples()
            current_word = trigrams_cpd[index].generate()
        else:
            break

    trigram_random_sentence += sentence_ending
    print("tri-grams random sentence : ")
    print(trigram_random_sentence)

    # ---------------------------- trigrams: End -----------------------------

main()
