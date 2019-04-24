from collections import defaultdict
import math
from aer import AERSufficientStatistics
from tqdm import tqdm, trange

class IBM1():
    def __init__(self, vocab_target, translation_probabilities = None):
        self.vocab_target = vocab_target        
        if translation_probabilities is None:
            default_probability = 1/len(vocab_target)
            self.translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability))
        else:
            self.translation_probabilities = translation_probabilities

    # Calculate log-likelihood of entire corpus
    def total_log_likelihood(self, corpus):
        print('\nCalculating log-likelihood')
        return sum(map(self.pair_log_likelihood, tqdm(corpus, desc='corpus')))

    def pair_log_likelihood(self, pair):
        # Expand sentence pair
        target_sentence, source_sentence = pair

        # Add null token
        source_sentence = [None] + source_sentence

        # Calculate log_likelihood of pair
        log_likelihood, _ = self.log_likelihood(target_sentence, source_sentence)
        return log_likelihood

    # Calculate target_token-likelihoods and log-likelihood of sentence pair
    def log_likelihood(self, target_sentence, source_sentence):    
        log_likelihood = -math.log(len(source_sentence)**len(target_sentence))
        target_likelihoods = defaultdict(lambda: 0)
        for target_token in target_sentence:                
            for source_token in source_sentence:
                target_likelihoods[target_token] += self.translation_probabilities[target_token][source_token]
            log_likelihood += math.log(target_likelihoods[target_token])
        return (log_likelihood, target_likelihoods)

    # Train model
    def train(self, training_corpus, iterations, validation_corpus, validation_gold):
        total_log_likelihoods = []
        aer_scores = []

        for i in trange(iterations, desc='iteration', position=0):
            expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times target_token is connected to source_token 
            expected_total = defaultdict(lambda: 0) # Expected total connections for source_token
            total_log_likelihood = 0

            # Calculate expected counts (Expectation step) and log-likelihood
            for t in trange(len(training_corpus), desc='training_corpus', position=1):
                # Expand sentence pair
                target_sentence, source_sentence = training_corpus[t]

                # Add null token
                source_sentence = [None] + source_sentence        

                # Calculate target_token-likelihoods and log-likelihood of sentence pair
                log_likelihood, target_likelihoods = self.log_likelihood(target_sentence, source_sentence)
                total_log_likelihood += log_likelihood

                # Collect counts
                for target_token in target_sentence:
                    for source_token in source_sentence:
                        normalized_count = self.translation_probabilities[target_token][source_token]/target_likelihoods[target_token]
                        expected_count[target_token][source_token] += normalized_count
                        expected_total[source_token] += normalized_count

            # Update translation probabilities (Maximization step)
            for target_token in expected_count.keys():
                for source_token in expected_count[target_token].keys():
                    self.translation_probabilities[target_token][source_token] = expected_count[target_token][source_token]/expected_total[source_token]

            print('\nIteration', i+1, 'complete')
            print('Log-likelihood before:', total_log_likelihood)
            aer = self.calculate_aer(validation_corpus, validation_gold)
            print('Validation AER after:', aer)
            total_log_likelihoods.append(total_log_likelihood)
            aer_scores.append(aer)
        return (total_log_likelihoods, aer_scores)

    # Print most likely translation for each foreign word
    def print_dictionary(self):
        for target_token in self.vocab_target:
            probs = self.translation_probabilities[target_token]
            print(target_token, max(zip(probs.values(), probs.keys())))

    # Find best alignment for a sentence pair
    def align(self, pair):
        # Expand sentence pair
        target_sentence, source_sentence = pair

        # Initialize alignment
        alignment = set()

        # Add best link for every english word
        for target_idx, target_token in enumerate(target_sentence):
            # Default alignment with null token
            best_align = None
            best_prob = self.translation_probabilities[target_token][None]     

            # Check alignments with all other possible words
            for source_idx, source_token in enumerate(source_sentence):
                prob = self.translation_probabilities[target_token][source_token] 
                if prob > best_prob:  # prefer newer word in case of tie
                    best_align = source_idx + 1 
                    best_prob = prob
            alignment.add((target_idx+1, best_align))

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
