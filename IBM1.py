from collections import defaultdict
import math
import aer

class IBM1():
	def __init__(self, english_vocab):
		self.english_vocab = english_vocab	
		default_probability = 1/len(english_vocab)
		self.translation_probabilities = defaultdict(lambda: defaultdict(lambda: default_probability))

	# TODO: this code is duplicated, make separate function for log-likelihood of one sentence pair
	def log_likelihood(self, training_corpus):
		print("\nCalculating log-likelihood")
		total_log_likelihood = 0
		for i in range(len(training_corpus)):
			# Expand sentence pair
			english_sentence, foreign_sentence = training_corpus[i]

			# Add null token
			foreign_sentence = [None] + foreign_sentence

			# Calculate e-likelihoods and log-likelihood of sentence pair
			log_likelihood = -math.log(len(foreign_sentence)**len(english_sentence))
			for e in english_sentence:
				e_likelihood = 0 				
				for f in foreign_sentence:
					e_likelihood += self.translation_probabilities[e][f]
				log_likelihood += math.log(e_likelihood)
			total_log_likelihood += log_likelihood

			# Print progress through training data
			if (i+1)%10000 == 0:
				print((i+1), "out of", len(training_corpus), "done")
		return total_log_likelihood
	
	# Train model
	def train(self, training_corpus, iterations, validation_corpus, validation_gold):
		for i in range(iterations):
			print("\nStarting iteration", i+1)
			expected_count = defaultdict(lambda: defaultdict(lambda: 0)) # Expected number of times e is connected to f 
			expected_total = defaultdict(lambda: 0) # Expected total connections for f
			total_log_likelihood = 0
			
			# Calculate expected counts (Expectation step) and log-likelihood
			for i in range(len(training_corpus)):
				# Expand sentence pair
				english_sentence, foreign_sentence = training_corpus[i]

				# Add null token
				foreign_sentence = [None] + foreign_sentence		

				# Calculate e-likelihoods and log-likelihood of sentence pair
				log_likelihood = -math.log(len(foreign_sentence)**len(english_sentence))
				e_likelihoods = defaultdict(lambda: 0)
				for e in english_sentence:				
					for f in foreign_sentence:
						e_likelihoods[e] += self.translation_probabilities[e][f]
					log_likelihood += math.log(e_likelihoods[e])
				total_log_likelihood += log_likelihood

				# Collect counts
				for e in english_sentence:
					for f in foreign_sentence:
						normalized_count = self.translation_probabilities[e][f]/e_likelihoods[e]
						expected_count[e][f] += normalized_count
						expected_total[f] += normalized_count

				# Print progress through training data
				if (i+1)%10000 == 0:
					print((i+1), "out of", len(training_corpus), "done")

			# Update translation probabilities (Maximization step)
			for e in expected_count.keys():
				for f in expected_count[e].keys():
					self.translation_probabilities[e][f] = expected_count[e][f]/expected_total[f]
				
			print("\nIteration", i+1, "complete")
			print("Log-likelihood:", total_log_likelihood)
			print("Validation AER:", self.calculate_aer(validation_corpus, validation_gold))

	# Print most likely translation for each foreign word
	def print_dictionary(self):
		for e in self.english_vocab:
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
					best_align = f_idx
					best_prob = prob
			alignment.add((e_idx, best_align))

		return alignment

	# Calculate AER on validation corpus using gold standard
	def calculate_aer(self, validation_corpus, validation_gold):
		# Compute predictions
		predictions = []
		for pair in validation_corpus:
			predictions.append(self.align(pair))

		# Compute AER
		metric = aer.AERSufficientStatistics()
		for gold, pred in zip(validation_gold, predictions):
			metric.update(sure=gold[0], probable=gold[1], predicted=pred)
		return metric.aer()

def main():
	# Read in training data
	training_e = open("data/training/hansards.36.2.e", "r", encoding="utf8")
	training_f = open("data/training/hansards.36.2.f", "r", encoding="utf8")
	training_corpus = []
	english_vocab = set()
	foreign_vocab = set()
	for _ in range(231164): # Hardcoded so I can vary training set size and dont have to count lines in the file (max 231164)
		english_sentence = training_e.readline().split(" ")[:-1]
		for word in english_sentence:
			english_vocab.add(word)
		foreign_sentence = training_f.readline().split(" ")[:-1]
		for word in foreign_sentence:
			foreign_vocab.add(word)
		training_corpus.append((english_sentence, foreign_sentence))
	print("English vocabulary size:", len(english_vocab))
	print("Foreign vocabulary size:", len(foreign_vocab))

	# Read in validation data
	validation_e = open("data/validation/dev.e", "r", encoding="utf8")
	validation_f = open("data/validation/dev.f", "r", encoding="utf8")
	validation_corpus = []
	for line in validation_e: 
		english_sentence = line.split(" ")[:-1]
		foreign_sentence = validation_f.readline().split(" ")[:-1]
		validation_corpus.append((english_sentence, foreign_sentence))
	validation_gold = aer.read_naacl_alignments("data/validation/dev.wa.nonullalign")

	# Create ibm1 model
	ibm1_model = IBM1(english_vocab)

	# Print initial log-likelihood and validation AER
	print("\nInitial log-likelihood:", ibm1_model.log_likelihood(training_corpus))
	print("Inital validation AER:", ibm1_model.calculate_aer(validation_corpus, validation_gold))
	
	# Train the model
	ibm1_model.train(training_corpus, 10, validation_corpus, validation_gold)

	# Print dictionary
	#ibm1_model.print_dictionary()
	
if __name__ == '__main__':
	main()