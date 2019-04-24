from collections import defaultdict
import math
from aer import AERSufficientStatistics
from tqdm import tqdm, trange

class IBM1():
    def __init__(self, vocab_en, translation_probabilities = None):
        self.vocab_en = vocab_en        
        if translation_probabilities is None:
            default_probability = 1/len(vocab_en)
            self.translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability))
        else:
            self.translation_probabilities = translation_probabilities

    # Calculate log-likelihood of entire corpus
    def total_log_likelihood(self, corpus):
        print('\nCalculating log-likelihood')
        return sum(map(self.pair_log_likelihood, tqdm(corpus, desc='corpus')))

    def pair_log_likelihood(self, pair):
        # Expand sentence pair
        english_sentence, foreign_sentence = pair

        # Add null token
        foreign_sentence = [None] + foreign_sentence

        # Calculate log_likelihood of pair
        log_likelihood, _ = self.log_likelihood(english_sentence, foreign_sentence)
        return log_likelihood

    # Calculate e-likelihoods and log-likelihood of sentence pair
    def log_likelihood(self, english_sentence, foreign_sentence):    
        log_likelihood = -math.log(len(foreign_sentence)**len(english_sentence))
        e_likelihoods = defaultdict(lambda: 0)
        for e in english_sentence:                
            for f in foreign_sentence:
                e_likelihoods[e] += self.translation_probabilities[e][f]
            log_likelihood += math.log(e_likelihoods[e])
        return (log_likelihood, e_likelihoods)

    # Train model
    def train(self, training_corpus, iterations, validation_corpus, validation_gold):
        total_log_likelihoods = []
        aer_scores = []

        for i in trange(iterations, desc='iterations', position=0):
            expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
            expected_total = defaultdict(lambda: 0) # Expected total connections for f
            total_log_likelihood = 0

            # Calculate expected counts (Expectation step) and log-likelihood
            for t in trange(len(training_corpus), desc='training_corpus', position=1):
                # Expand sentence pair
                english_sentence, foreign_sentence = training_corpus[t]

                # Add null token
                foreign_sentence = [None] + foreign_sentence        

                # Calculate e-likelihoods and log-likelihood of sentence pair
                log_likelihood, e_likelihoods = self.log_likelihood(english_sentence, foreign_sentence)
                total_log_likelihood += log_likelihood

                # Collect counts
                for e in english_sentence:
                    for f in foreign_sentence:
                        normalized_count = self.translation_probabilities[e][f]/e_likelihoods[e]
                        expected_count[e][f] += normalized_count
                        expected_total[f] += normalized_count

            # Update translation probabilities (Maximization step)
            for e in expected_count.keys():
                for f in expected_count[e].keys():
                    self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]

            print('\nIteration', i+1, 'complete')
            print('Log-likelihood before:', total_log_likelihood)
            aer = self.calculate_aer(validation_corpus, validation_gold)
            print('Validation AER after:', aer)
            total_log_likelihoods.append(total_log_likelihood)
            aer_scores.append(aer)
        return (total_log_likelihoods, aer_scores)

    # Print most likely translation for each foreign word
    def print_dictionary(self):
        for e in self.vocab_en:
            probs = self.translation_probabilities[e]
            print(e, max(zip(probs.values(), probs.keys())))

    # Find best alignment for a sentence pair
    def align(self, pair):
        # Expand sentence pair
        english_sentence, foreign_sentence = pair

        # Initialize alignment
        alignment = set()

        # Add best link for every english word
        for e_idx, e in enumerate(english_sentence):
            # Default alignment with null token
            best_align = None
            best_prob = self.translation_probabilities[e][None]     

            # Check alignments with all other possible words
            for f_idx, f in enumerate(foreign_sentence):
                prob = self.translation_probabilities[e][f] 
                if prob > best_prob:  # prefer newer word in case of tie
                    best_align = f_idx + 1 
                    best_prob = prob
            alignment.add((e_idx+1, best_align))

        return alignment

    # Calculate AER on validation corpus using gold standard
    def calculate_aer(self, validation_corpus, validation_gold):
        # Compute predictions
        predictions = []
        for pair in validation_corpus:
            predictions.append(self.align(pair))

        # Compute AER
        metric = AERSufficientStatistics()
        for gold, pred in zip(validation_gold, predictions):
            metric.update(sure=gold[0], probable=gold[1], predicted=pred)
        return metric.aer()
