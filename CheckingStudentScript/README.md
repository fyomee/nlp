Given we have:
 * a fill-in the gap question file *question.in* that contains a text with some gaps,
 * a student script that contains the student answers *script.in* file with the gaps in the text filled, and
 * the *answer.in* file that contains the list of tokens to be placed in the gaps in the order of their appearance in *question.in*. Those tokens are stemmed using the Porter's stemmer.

Write a Python program *scriptChecker.py* that checks the student script. Program should extract from the student script the answers and compare them with the list of tokens in the *answer.in* file.
