import nltk
import pandas


def learn_naive_bayes_text(path='IMDB', targets=('pos', 'neg'),
                           low=5, high=24004):
    vocabulary = set()
    target_probabilities = {}
    tokens = {}
    for target in targets:
        target_probabilities[target] = 1.0 / len(targets)
        tokens[target] = []
        for value in range(low, high+1):
            text = nltk.data.load('{}/{}/{}.txt'.format(path, target, value),
                                  format='text')
            text_tokens = nltk.word_tokenize(text)
            tokens[target] += text_tokens
            vocabulary.update(text_tokens)
    token_probabilities = pandas.DataFrame(index=vocabulary, columns=targets)
    for target in targets:
        text = nltk.Text(tokens[target])
        target_denominator = len(tokens[target]) + len(vocabulary)
        token_probabilities[:, target] = pandas.Series(
            {token: (text.count(token)+1) / target_denominator
             for token in vocabulary})
    return target_probabilities, token_probabilities
