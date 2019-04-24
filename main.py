from collections import defaultdict
import argparse
import itertools
from typing import Tuple, List, Set, Dict, DefaultDict
import matplotlib.pyplot as plt
import aer
from tqdm import tqdm, trange
from pdb import set_trace
from ibm1 import IBM1
from ibm2 import IBM2
from jump import IBM2Jump
import dill as pickle

def get_flags():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = 'ibm1', help='model, ibm1 (default), ibm2, jump')
    parser.add_argument('--lines', type = int, default = None, help='number of lines of training data to use, default all')
    parser.add_argument('--iterations', type = int, default = 15, help='number of iterations, default 15')
    parser.add_argument('--probabilities', type = str, default = None, help='file to load previously trained probabilities from (default none)')
    flags, unparsed = parser.parse_known_args()
    return flags

def get_model(model: str, vocab_target: Set[str], probabilities: DefaultDict[str, DefaultDict[str, int]]):
    if model == 'ibm2':
        ibm_model = IBM2(vocab_target, probabilities)
    if model == 'jump':
        ibm_model = IBM2Jump(vocab_target, probabilities)
    else:
        ibm_model = IBM1(vocab_target, probabilities)
    return ibm_model

def read_tokens(path: str, n:int=None) -> List[List[str]]:
    sentences = open(path, 'r', encoding='utf8').readlines()
    sentences_ = itertools.islice(sentences, n)
    # we split on spaces as the hansards dataset uses explicit spacing between tokens
    return [sentence.split(' ')[:-1] for sentence in sentences_]

def sentence_vocab(tokenized: List[List[str]]) -> Set[str]:
    return set([token for tokens in tokenized for token in tokens])

def read_data(n:int=None):
    # Read in training data
    tokenized_target = read_tokens('data/training/hansards.36.2.e', n)
    tokenized_source = read_tokens('data/training/hansards.36.2.f', n)
    training_corpus = list(zip(tokenized_target, tokenized_source))
    vocab_target = sentence_vocab(tokenized_target)
    print(f'vocabulary size english: {len(vocab_target)}')

    # Read in validation data
    validation_corpus = list(zip(
        read_tokens('data/validation/dev.e'),
        read_tokens('data/validation/dev.f'),
    ))
    validation_gold = aer.read_naacl_alignments('data/validation/dev.wa.nonullalign')

    # Read in test data
    test_corpus = list(zip(
        read_tokens('data/testing/test/test.e'),
        read_tokens('data/testing/test/test.f'),
    ))
    test_gold = aer.read_naacl_alignments('data/testing/answers/test.wa.nonullalign')

    return (training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, vocab_target)

def test_model(ibm_model, training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, iterations) -> None:
    # Print initial validation AER    
    initial_aer = ibm_model.calculate_aer(validation_corpus, validation_gold)
    print('Initial validation AER:', initial_aer)

    # Train the model
    log_likelihoods, aer_scores = ibm_model.train(training_corpus, iterations, test_corpus, test_gold)
    # aer_scores = [initial_aer] + aer_scores

    # Print log-likelihood after training
    final_log_likelihood = ibm_model.total_log_likelihood(training_corpus)
    print('\nFinal log-likelihood:', final_log_likelihood)
    # TODO: have train() return the log likelihoods from after the iteration as well?
    log_likelihoods = [*log_likelihoods[1:], final_log_likelihood]
    xs = list(range(1, iterations+1))
    # TODO: tensorboard?
    stats = {
        'Log-likelihood': log_likelihoods,
        'AER Score': aer_scores,
    }
    # Plot log-likelihood and aer curves
    for label, ys in stats.items():
        # TODO: seaborn?
        fig = plt.figure()
        plt.plot(xs, ys)
        plt.xlabel('Iterations')
        plt.ylabel(label)
        plt.savefig('IBM1_log_likelihoods.png')
        plt.close(fig)

    # TODO: df.to_csv?
    f = open('run_log.txt','w+')
    for i in tqdm(range(iterations), desc='writing'):
        f.write('Iteration: ' + str(i+1) + ' Log-likelihood: ' + str(log_likelihoods[i]) + ' AER score: ' + str(aer_scores[i]) + '\n')

    # # Print dictionary
    # ibm_model.print_dictionary()

def main():
    flags = get_flags()
    (training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, vocab_target) = read_data(flags.lines)

    # optionally load in previously trained probabilities
    prob_f = flags.probabilities
    if prob_f:
        with open(prob_f, 'rb') as f:
            probabilities = pickle.load(f)

    ibm_model = get_model(flags.model, vocab_target, probabilities)
    with open(f'{flags.model}.pkl', 'wb') as f:
        pickle.dump(ibm_model.translation_probabilities, f)
    test_model(ibm_model, training_corpus, validation_corpus, test_corpus, validation_gold, test_gold, flags.iterations)

if __name__ == '__main__':
    main()
