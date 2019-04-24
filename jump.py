from collections import defaultdict
import math
from ibm1 import IBM1

class IBM2Jump(IBM1):
    def __init__(self, english_vocab, translation_probabilities = None):
        super(IBM2Jump, self).__init__(english_vocab, translation_probabilities)
        self.jump_counts = defaultdict(lambda: 1)

    # Calculate e-likelihoods and log-likelihood of sentence pair
    def log_likelihood(self, english_sentence, foreign_sentence):    
        len_e = len(english_sentence)
        len_f = len(foreign_sentence)
        log_likelihood = 0
        e_likelihoods = defaultdict(lambda: 0)

        for e_idx, e in enumerate(english_sentence):    
            normalizer = 0 # Normalize to have sensible log_likelihood
            for f_idx, f in enumerate(foreign_sentence):
                normalizer += self.jump_counts[f_idx - int(e_idx*len_f/len_e)]
            for f_idx, f in enumerate(foreign_sentence):
                translation_probability = self.translation_probabilities[e][f]
                alignment_probabilility = self.jump_counts[f_idx - int(e_idx*len_f/len_e)]/normalizer
                e_likelihoods[e] += translation_probability*alignment_probabilility
            log_likelihood += math.log(e_likelihoods[e])
        return (log_likelihood, e_likelihoods)

    # Train model
    def train(self, training_corpus, iterations, validation_corpus, validation_gold):
        total_log_likelihoods = []
        aer_scores = []
        for i in range(iterations):
            print("\nStarting iteration", i+1)
            expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
            expected_total = defaultdict(lambda: 0) # Expected total connections for f
            expected_jump_count = defaultdict(lambda: 0) # Expected connections with a certain jump length
            total_log_likelihood = 0

            # Calculate expected counts (Expectation step) and log-likelihood
            for t in range(len(training_corpus)):
                # Expand sentence pair
                english_sentence, foreign_sentence = training_corpus[t]

                # Add null token
                foreign_sentence = [None] + foreign_sentence

                len_e = len(english_sentence)
                len_f = len(foreign_sentence)

                # Calculate e-likelihoods and log-likelihood of sentence pair
                log_likelihood, e_likelihoods = self.log_likelihood(english_sentence, foreign_sentence)
                total_log_likelihood += log_likelihood

                # Collect counts
                for e_idx, e in enumerate(english_sentence):
                    normalizer = 0
                    for f_idx, f in enumerate(foreign_sentence):
                        normalizer += self.jump_counts[f_idx - int(e_idx*len_f/len_e)]
                    for f_idx, f in enumerate(foreign_sentence):
                        translation_probability = self.translation_probabilities[e][f]
                        alignment_probabilility = self.jump_counts[f_idx - int(e_idx*len_f/len_e)]/normalizer
                        normalized_count = translation_probability*alignment_probabilility/e_likelihoods[e]
                        expected_count[e][f] += normalized_count
                        expected_total[f] += normalized_count
                        expected_jump_count[f_idx - int(e_idx*len_f/len_e)] += normalized_count

                # Print progress through training data
                if (t+1)%10000 == 0:
                    print((t+1), "out of", len(training_corpus), "done")

            # Update translation probabilities (Maximization step)
            for e in expected_count.keys():
                for f in expected_count[e].keys():
                    self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]

            # Update jump counts (Maximization step)
            for jump in expected_jump_count.keys():
                self.jump_counts[jump] = expected_jump_count[jump]

            aer = self.calculate_aer(validation_corpus, validation_gold)
            print("\nIteration", i+1, "complete")
            print("Log-likelihood before:", total_log_likelihood)
            print("Validation AER after:", aer)
            total_log_likelihoods.append(total_log_likelihood)
            aer_scores.append(aer)
        return (total_log_likelihoods, aer_scores)

    # Find best alignment for a sentence pair
    def align(self, pair):
        # Expand sentence pair
        english_sentence, foreign_sentence = pair
        len_e = len(english_sentence)
        len_f = len(foreign_sentence)

        # Initialize alignment
        alignment = set()

        # Add best link for every english word
        for e_idx, e in enumerate(english_sentence):
            # Default alignment with null token
            best_align = None
            best_prob = self.translation_probabilities[e][None]*self.jump_counts[0 - int(e_idx*len_f/len_e)] 

            # Check alignments with all other possible words
            for f_idx, f in enumerate(foreign_sentence):
                prob = self.translation_probabilities[e][f]*self.jump_counts[f_idx - int(e_idx*len_f/len_e)] 
                if prob > best_prob:  # prefer newer word in case of tie
                    best_align = f_idx + 1 
                    best_prob = prob
            alignment.add((e_idx+1, best_align))

        return alignment
