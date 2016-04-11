import nltk
import pandas


def learn_naive_bayes_text(targets=('pos', 'neg')):
    vocabulary = set()
    target_probabilities = {}
    tokens = {}
    for target in targets:
        # todos conjuntos possuem o mesmo tamanho
        target_probabilities[target] = 1.0 / len(targets)
        text = nltk.data.load('IMDB/{}.txt'.format(target), format='text')
        text_tokens = nltk.wordpunct_tokenize(text)
        tokens[target] = text_tokens
        vocabulary.update(text_tokens)
    token_probabilities = pandas.DataFrame(index=vocabulary, columns=targets)
    for target in targets:
        frequencies = nltk.FreqDist(tokens[target])
        token_probabilities.loc[frequencies, target] = frequencies
        token_probabilities.loc[frequencies, target] += 1
        token_probabilities.loc[frequencies, target] /= \
            len(tokens[target]) + len(vocabulary)
    token_probabilities = token_probabilities.fillna(0)
    return target_probabilities, token_probabilities
