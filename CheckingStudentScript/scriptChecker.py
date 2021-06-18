import re

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def main():
    with open ('question.in') as questions:
        questionTokens = word_tokenize(questions.read())

    with open ('script.in') as scripts:
        scriptTokens = word_tokenize(scripts.read())

    # with open ('answer.in') as answers:
    #     answerTokens = word_tokenize(answers.read())

    file = open('answer.in')
    answers = []
    for word in file:
        answers.append(word[:-1])

    porter = PorterStemmer()

    # We can check word by word and find what missing in question.in

    # studentAnswers=[]
    # for i in range(len(scriptTokens)):
    #     if scriptTokens[i] != questionTokens[i]:
    #         studentAnswers.append(porter.stem(scriptTokens[i]))
    #
    # point = 0;
    # for i in range(len(studentAnswers)):
    #     if answers[i] == studentAnswers[i]:
    #         point = point + 1


    # Or if we can consider the question patterns will always be the same as given example where gaps represented by number of _.
    # Then, since words normally don't have two consecutive _ so we can write a regular expression considering it will be 2 or
    # more underscores which will define a gap and using this we can find the indexes which can be used later for finding the
    # answers from scripts

    gapsIdx = [i for i, questionToken in enumerate(questionTokens) if re.search('_{2,}', questionToken)]

    point = 0;
    for (i, idx) in zip(range(len(gapsIdx)), gapsIdx):
        if answers[i] == porter.stem(scriptTokens[idx]):
            point = point + 1

    print("Total Point : ", point)

main()
