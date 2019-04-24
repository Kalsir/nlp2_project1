from collections import defaultdict
import math
from ibm1 import IBM1

class AlignmentProbabilities():
    def __init__(self):
        self.alignment_probabilities = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))

    def read(self, len_e, len_f, e_idx, f_idx):
        if f_idx not in self.alignment_probabilities[len_e][len_f][e_idx]:
            self.alignment_probabilities[len_e][len_f][e_idx][f_idx] = 1/10
        return self.alignment_probabilities[len_e][len_f][e_idx][f_idx]

    def write(self, len_e, len_f, e_idx, f_idx, value):
        self.alignment_probabilities[len_e][len_f][e_idx][f_idx] = value

class IBM2(IBM1):
    def __init__(self, vocab_en, translation_probabilities = None):
        super(IBM2, self).__init__(vocab_en, translation_probabilities)
        self.alignment_probabilities = AlignmentProbabilities()

    # Calculate e-likelihoods and log-likelihood of sentence pair
    def log_likelihood(self, english_sentence, foreign_sentence):    
        len_e = len(english_sentence)
        len_f = len(foreign_sentence)
        log_likelihood = 0
        e_likelihoods = defaultdict(lambda: 0)
        for e_idx, e in enumerate(english_sentence):                
            for f_idx, f in enumerate(foreign_sentence):
                translation_probability = self.translation_probabilities[e][f]
                alignment_probabilility = self.alignment_probabilities.read(len_e, len_f, e_idx, f_idx)
                e_likelihoods[e] += translation_probability*alignment_probabilility
            log_likelihood += math.log(e_likelihoods[e])
        return (log_likelihood, e_likelihoods)

    # Train model
    def train(self, training_corpus, iterations, validation_corpus, validation_gold):
        total_log_likelihoods = []
        aer_scores = []
        for i in tqdm(range(iterations)):
            print('\nStarting iteration', i+1)
            expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
            expected_total = defaultdict(lambda: 0) # Expected total connections for f
            len_expected_count = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))) # Expected number of times e is connected to f in pairs of specific length
            len_expected_total = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0))) # Expected number of total connections for f in pairs of specific length
            total_log_likelihood = 0

            # Calculate expected counts (Expectation step) and log-likelihood
            for t in range(len(training_corpus)):
                # Expand sentence pair
                english_sentence, foreign_sentence = training_corpus[t]

                # Add null token
                foreign_sentence = [None] + foreign_sentence

                # Calculate sentence lenghts 
                len_e = len(english_sentence)
                len_f = len(foreign_sentence)                    

                # Calculate e-likelihoods and log-likelihood of sentence pair
                log_likelihood, e_likelihoods = self.log_likelihood(english_sentence, foreign_sentence)
                total_log_likelihood += log_likelihood

                # Collect counts
                for e_idx, e in enumerate(english_sentence):
                    for f_idx, f in enumerate(foreign_sentence):
                        translation_probability = self.translation_probabilities[e][f]
                        alignment_probabilility = self.alignment_probabilities.read(len_e, len_f, e_idx, f_idx)
                        normalized_count = translation_probability*alignment_probabilility/e_likelihoods[e]
                        expected_count[e][f] += normalized_count
                        expected_total[f] += normalized_count
                        len_expected_count[len_e][len_f][e_idx][f_idx] += normalized_count
                        len_expected_total[len_e][len_f][f_idx] += normalized_count

                # Print progress through training data
                if (t+1)%10000 == 0:
                    print((t+1), 'out of', len(training_corpus), 'done')

            # Update translation probabilities (Maximization step)
            for e in expected_count.keys():
                for f in expected_count[e].keys():
                    self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]

            # Update alignment probabilities (Maximization step)
            for len_e in len_expected_count.keys():
                for len_f in len_expected_count[len_e].keys():
                    for e_idx in len_expected_count[len_e][len_f].keys():
                        for f_idx in len_expected_count[len_e][len_f][e_idx].keys():
                            new_value = len_expected_count[len_e][len_f][e_idx][f_idx]/len_expected_total[len_e][len_f][f_idx]
                            self.alignment_probabilities.write(len_e, len_f, e_idx, f_idx, new_value)

            print('\nIteration', i+1, 'complete')
            print('Log-likelihood before:', total_log_likelihood)
            aer = self.calculate_aer(validation_corpus, validation_gold)
            print('Validation AER after:', aer)
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
            best_prob = self.translation_probabilities[e][None]*self.alignment_probabilities.read(len_e, len_f, e_idx, 0) 

            # Check alignments with all other possible words
            for f_idx, f in enumerate(foreign_sentence):
                prob = self.translation_probabilities[e][f]*self.alignment_probabilities.read(len_e, len_f, e_idx, f_idx)  
                if prob > best_prob:  # prefer newer word in case of tie
                    best_align = f_idx + 1 
                    best_prob = prob
            alignment.add((e_idx+1, best_align))

        return alignment
