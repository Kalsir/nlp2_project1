import os
from collections import defaultdict
import argparse
import itertools
from typing import Tuple, List, Set, Dict, DefaultDict
from operator import itemgetter
import matplotlib.pyplot as plt
import aer
from tqdm import tqdm, trange
from pdb import set_trace
from ibm1 import IBM1
from ibm2 import IBM2
from jump import IBM2Jump
import dill as pickle
import numpy as np

def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'ibm1', help='model, ibm1 (default), ibm2, jump')
    parser.add_argument('--lines', type = int, default = None, help='number of lines of training data to use, default all')
    parser.add_argument('--seed', type = int, default = 42, help='random seed, default 42')
    parser.add_argument('--iterations', type = int, default = 10, help='number of iterations, default 15')
    parser.add_argument('--probabilities', type = str, default = None, help='file to load previously trained probabilities from (default none)')
    parser.add_argument('--sampling_method', type = str, default = 'uniform', help='sampling method for initial probabilities: uniform (default), random')
    parser.add_argument('--lower', action='store_true', help='lowercase tokens')
    flags, unparsed = parser.parse_known_args()
    return flags

def get_model(model: str, vocab_target: Set[str], probabilities: DefaultDict[str, DefaultDict[str, int]], sampling_method: str = 'uniform', seed=42):
    args = [vocab_target, probabilities, sampling_method, seed]
    if model == 'ibm2':
        ibm_model = IBM2(*args)
    elif model == 'jump':
        ibm_model = IBM2Jump(*args)
    else:
        ibm_model = IBM1(*args)
    return ibm_model

def read_tokens(path: str, lower=False, n:int=None) -> List[List[str]]:
    sentences = open(path, 'r', encoding='utf8').readlines()
    sentences_ = itertools.islice(sentences, n)
    # we split on spaces as the hansards dataset uses explicit spacing between tokens
    return [[token.lower() if lower else token for token in sentence.split(' ')[:-1]] for sentence in sentences_]

def sentence_vocab(tokenized: List[List[str]]) -> Set[str]:
    return set([token for tokens in tokenized for token in tokens])

def read_data(n:int=None, lower=False):
    # Read in training data
    tokenized_target = read_tokens('../data/training/hansards.36.2.f', lower, n)
    tokenized_source = read_tokens('../data/training/hansards.36.2.e', lower, n)
    training_corpus = list(zip(tokenized_target, tokenized_source))
    vocab_target = sentence_vocab(tokenized_target)
    print(f'vocabulary size english: {len(vocab_target)}')

    # Read in validation data
    validation_corpus = list(zip(
        read_tokens('../data/validation/dev.f', lower),
        read_tokens('../data/validation/dev.e', lower),
    ))
    validation_gold = aer.read_naacl_alignments('../data/validation/dev.wa.nonullalign')

    # Read in test data
    test_corpus = list(zip(
        read_tokens('../data/testing/test/test.f', lower),
        read_tokens('../data/testing/test/test.e', lower),
    ))
    test_gold = aer.read_naacl_alignments('../data/testing/answers/test.wa.nonullalign')

    return (training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, vocab_target)

def test_model(ibm_model, training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, iterations, name = None) -> None:
    if not name:
        name = ibm_model.__class__.__name__        
    # Print initial validation AER
    initial_aer = ibm_model.calculate_aer(validation_corpus, validation_gold)
    print('Initial validation AER:', initial_aer)

    # Train the model
    log_likelihoods, aer_scores = ibm_model.train(training_corpus, iterations, validation_corpus, validation_gold, test_corpus, test_gold, name)
    # aer_scores = [initial_aer] + aer_scores

    # Print log-likelihood after training
    # final_log_likelihood = ibm_model.total_log_likelihood(training_corpus)
    final_log_likelihood = ibm_model.total_log_likelihood(training_corpus) # Added dividing in ibm2 model itself /len(training_corpus)
    print('\nFinal log-likelihood:', final_log_likelihood)
    # TODO: have train() return the log likelihoods from after the iteration as well?
    log_likelihoods = [*log_likelihoods[1:], final_log_likelihood/len(training_corpus)]
    xs = list(range(1, iterations+1))
    # TODO: tensorboard?
    stats = {
        'logp': log_likelihoods,
        'aer': aer_scores,
    }
    labels = {
        'logp': 'Average log-likelihood',
        'aer': 'AER Score',
    }
    # Plot log-likelihood and aer curves
    for i, (k, ys) in enumerate(stats.items()):
        # TODO: seaborn?
        label = labels[k]
        png_file = os.path.join('..', 'Results', 'plots', f'{k}_{name}.png')
        fig = plt.figure()
        plt.plot(xs, ys)
        plt.xlabel('Iterations')
        plt.ylabel(label)
        plt.savefig(png_file)
        plt.close(fig)

    # TODO: df.to_csv?
    log_file = os.path.join('..', 'Results', 'logs', f'run_log_{name}.txt')
    with open(log_file,'w+') as f:
        for i in tqdm(range(iterations), desc='writing'):
            f.write('Iteration: ' + str(i+1) + ' Log-likelihood: ' + str(log_likelihoods[i]) + ' AER score: ' + str(aer_scores[i]) + '\n')

    # # Print dictionary
    # ibm_model.print_dictionary()

def main():
    flags = get_flags()
    flag_keys = ['model', 'lines', 'iterations', 'probabilities', 'sampling_method', 'lower', 'seed']
    (model, lines, iterations, probabilities, sampling_method, lower, seed) = itemgetter(*flag_keys)(vars(flags))
    name = f'{model}-{sampling_method}'
    if seed != 42:
        name += f'-{seed}'
    if lower:
        name += f'-lower'
    if probabilities:
        name += f'-{probabilities}'

    (training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, vocab_target) = read_data(lines, lower)

    # optionally load in previously trained probabilities
    if probabilities:
        with open(probabilities, 'rb') as f:
            translation_probabilities = pickle.load(f)
    else:
        if sampling_method == 'random':
            np.random.seed(seed)
            translation_probabilities = dict()
            for pair in training_corpus:
                target, source = pair
                source = [None] + source
                for t in target:
                    if t not in translation_probabilities.keys():
                        translation_probabilities[t] = dict()
                    for s in source:
                        translation_probabilities[t][s] = np.random.rand()*2/len(vocab_target)
            for pair in validation_corpus:
                target, source = pair
                source = [None] + source
                for t in target:
                    if t not in translation_probabilities.keys():
                        translation_probabilities[t] = dict()
                    for s in source:
                        translation_probabilities[t][s] = np.random.rand()*2/len(vocab_target)
            for pair in test_corpus:
                target, source = pair
                source = [None] + source
                for t in target:
                    if t not in translation_probabilities.keys():
                        translation_probabilities[t] = dict()
                    for s in source:
                        translation_probabilities[t][s] = np.random.rand()*2/len(vocab_target)
        else:
            translation_probabilities = None

    ibm_model = get_model(model, vocab_target, translation_probabilities, sampling_method, seed)

    test_model(ibm_model, training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, iterations, name)
    pkl_file = os.path.join('..', 'Results', 'pickles', f'{name}.pkl')
    with open(pkl_file, 'wb') as f:
        pickle.dump(ibm_model.translation_probabilities, f)

if __name__ == '__main__':
    main()
