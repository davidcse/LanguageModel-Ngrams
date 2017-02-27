from sys import argv
import math
from functools import reduce

# Dictionaries storing file probability data.
unigram_data_dict = {}
bigram_data_dict = {}

# Uses the perplexity formula, provided that you give it a calculated sum probability component for the formula.
# Also, needs the sequence length, N, to take the nth root.
def perplexity_formula(probability_sum, sequence_length):
    inverse_seq = -1/sequence_length
    return 2**(inverse_seq * probability_sum)

# calculates the laplace probability for a single word.
# uses global variables that are calculations derived from the
# language model files that were opened and parsed.
def unigram_laplace_prob_calculation(word):
    global unigram_data_dict
    global total_token
    vocabulary_size = len(list(unigram_data_dict.keys()))
    if word in unigram_data_dict:
        unigram_data = unigram_data_dict[word]
        count_addone = int(unigram_data[1]) + 1
        unigram_laplace_prob = count_addone/(total_token + vocabulary_size + 1)
    else:
        unigram_laplace_prob = 1/(total_token + vocabulary_size + 1)
    return unigram_laplace_prob


# Open the file and read the data in the file. Also take user input choice.
try:
    script, bigram_file, unigram_file, test_file = argv
except ValueError:
    print("Please run program with following format : 1)bigram.lm  2)unigram.lm  3)test file as parameters via the commandline")
    exit()


######## READ DATA FILES AND CREATE LOOKUP MAPS ########
bigram_file = open(bigram_file.strip(),"r")
unigram_file = open(unigram_file.strip(),"r")
bigram_prob_data = bigram_file.read().split("\n")
unigram_prob_data = unigram_file.read().split("\n")
bigram_file.close()
unigram_file.close()

# parse data line by line for bigram.lm
for i in range(len(bigram_prob_data)):
    line = bigram_prob_data[i].split()
    if(len(line) == 0): # skip empty lines
        continue
    bigram_data_dict[(line[0],line[1])] = line

# parse data line by line for unigram.lm
for i in range(len(unigram_prob_data)):
    line = unigram_prob_data[i].split()
    if(len(line) == 0):
        continue
    unigram_data_dict[line[0]] = line


#### GET TEST DATA
test_file = open(test_file.strip(),"r")
test_corpus = test_file.read().split("\n")
test_file.close()

### REMOVE EMPTY LINES FROM TEST DATA
processed_test_corpus = []
for i in range(len(test_corpus)):
    if(test_corpus[i].strip() == ""):
        continue
    else:
        processed_test_corpus.append(test_corpus[i])

### INSERT START AND END TOKENS
for i in range(len(processed_test_corpus)):
    processed_test_corpus[i] = processed_test_corpus[i].split()
    processed_test_corpus[i].insert(0,"<s>")
    processed_test_corpus[i].append("</s>")

#### CREATE SINGLE SEQUENCE
sequence = []
for sentence_list in processed_test_corpus:
    for token in sentence_list:
        sequence.append(token)


###### CALCULATE PERPLEXITY
laplace_bigram_sum = 0
laplace_unigram_sum = 0
interpolation_bigram_sum = 0
vocabulary_size = len(list(unigram_data_dict.keys()))
training_lambda = 0.1
total_token = 0

# assign proper value to total_token
count_array = []
for i in unigram_data_dict:
    count_array.append(int(unigram_data_dict[i][1]))
total_token = reduce(lambda x,y : x + y, count_array)



##### CALCULATE BIGRAM PROBABILITIES SUMS
for i in range(len(sequence)-1):
    bigram_tuple = (sequence[i],sequence[i+1])
    if bigram_tuple in bigram_data_dict:
        bigram_laplace_data = float(bigram_data_dict[bigram_tuple][4])
        bigram_interpolated_data = float(bigram_data_dict[bigram_tuple][5])
    else:   #### calculate when not seen
        if sequence[i] in unigram_data_dict:
            count_first_word = int(unigram_data_dict[sequence[i]][1])
        else:
            count_first_word = 0
        bigram_laplace_data = 1 / (count_first_word + vocabulary_size + 1)
        bigram_interpolated_data = (1-training_lambda) * unigram_laplace_prob_calculation(sequence[i+1])
    laplace_bigram_sum = laplace_bigram_sum + math.log(bigram_laplace_data,2)
    interpolation_bigram_sum = interpolation_bigram_sum + math.log(bigram_interpolated_data,2)

#### CALCULATE UNIGRAM PROBABILITY SUMS
for i in range(len(sequence)):
    word = sequence[i]
    laplace_unigram_sum = laplace_unigram_sum + math.log(unigram_laplace_prob_calculation(word), 2)

####### OUTPUT THE FINAL PERPLEXITIES
laplace_bigram_perplexity = perplexity_formula(laplace_bigram_sum, len(sequence))
interpolation_perplexity = perplexity_formula(interpolation_bigram_sum, len(sequence))
laplace_unigram_perplexity = perplexity_formula(laplace_unigram_sum, len(sequence))
print("Laplace Bigram Perplexity : " + str(laplace_bigram_perplexity))
print("Interpolated Bigram Perplexity : " + str(interpolation_perplexity))
print("Laplace Unigram Perplexity : " + str(laplace_unigram_perplexity))
