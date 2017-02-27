from sys import argv
from functools import reduce
import math
import operator

# GLOBALS
bigram_tuple_dict = {}
unigram_dict = {}
sentences_array = []
joint_prob_array = []


# PREPROCESSING
# arg1 : corpus of text strings to break apart into list of word lists.
# arg2 : list to build. Fill with list of word tokens.
def build_sentences_array(text_corpus, output_array):
    corpus = text_corpus.split(".")
    for sentence in corpus :
        words_array = sentence.split()
        for i in range(len(words_array)):
            words_array[i] = words_array[i].lower()
        words_array.insert(0,"<s>")
        words_array.append("</s>")
        output_array.append(words_array)

# BUILD THE BIGRAM DICT.
# sentences_array : 2D array: list of sentences, where each sentence is a list of strings. [[str,str],[str,str]]
# bigram_tuple_dict : dictionary. key filled with bigram tuples(str,str). Value is count (int)
def build_bigram_tuple_dict(sentences_array, bigram_tuple_dict):
    for word_array in sentences_array :
        for i in range(len(word_array)-1):
            tup = (word_array[i],word_array[i+1])
            if tup in bigram_tuple_dict:
                bigram_tuple_dict[tup] = bigram_tuple_dict[tup] + 1
            else:
                bigram_tuple_dict[tup] = 1

# BUILD THE UNIGRAM DICT
# arg1 : 2D array, list of list of words
# arg2 : dictionary. key (str). Value is count (int)
def build_unigram_dict(sentences_array,unigram_dict):
    for word_array in sentences_array:
        for word in word_array:
            if word in unigram_dict:
                unigram_dict[word] = unigram_dict[word]+1
            else:
                unigram_dict[word] = 1

######  PROBABILITY FUNCTIONS #############
def probability_mle(word1,word2):
    global bigram_tuple_dict
    global unigram_dict
    # check in global dictionary for count(word1,word)
    if(((word1,word2) in bigram_tuple_dict) and (word1 in unigram_dict)):
        return bigram_tuple_dict[(word1,word2)] / unigram_dict[word1]
    else:
        return 0 # value error


# calculates the ad probabilities for Pr_AD(word2 | word 1)
def probability_ad(word1,word2):
    global bigram_tuple_dict
    global unigram_dict
    discount_value = 0.5
    # check in global dictionary for count(word1,word)
    if(((word1,word2) in bigram_tuple_dict) and (word1 in unigram_dict)):
        return (bigram_tuple_dict[(word1,word2)] - discount_value) / unigram_dict[word1]
    else:
        return 0 # value error

# calculates the laplace smoothed probabilities for Pr_L(word2 | word1)
def probability_laplace_bigram(word1,word2):
    global bigram_tuple_dict
    global unigram_dict
    vocabulary_size = len(list(unigram_dict.keys()))
    if (word1,word2) in bigram_tuple_dict:
        count_bigram = bigram_tuple_dict[(word1,word2)]
    else:
        count_bigram = 0
    if word1 in unigram_dict:
        count_unigram = unigram_dict[word1]
    else:
        count_unigram = 0
    # using formula:  count(x,y) + 1 / (count(x) + vocabulary_size + 1)
    return (count_bigram + 1) / (count_unigram + vocabulary_size + 1)


# calculates the laplace smoothed probabilities for Pr_L(word) using unigram data.
def probability_laplace_unigram(word):
    global unigram_dict
    vocabulary_size = len(list(unigram_dict.keys()))
    total_token_num = reduce(lambda x,y: x+y, unigram_dict.values()) # sum all the values in the unigram_dict
    if word in unigram_dict:
        count = unigram_dict[word]
        return (count + 1)/(total_token_num + vocabulary_size + 1)
    else:
        return 1/(total_token_num + vocabulary_size + 1)

# Breaks a corpus of text into one list sequence of tokens, delimited by newlines and spaces.
# Ideally, would be the list of words in the corpus
def split_into_sequence_array(dev_set_corpus):
    dev_set_sentences = dev_set_corpus.split("\n")
    dev_set_sequence = []
    for sent in dev_set_sentences:
        tokens = sent.split(" ")
        for t in tokens:
            dev_set_sequence.append(t)
    return dev_set_sequence


# only call after bigram.lm and  unigram.lm are finished calculating and processed.
def calculate_perplexity(dev_set_corpus):
    perplexities = [0]*5
    sequence = split_into_sequence_array(dev_set_corpus)
    for i in range(len(sequence)-1):
        word1 = sequence[i]
        word2 = sequence[i+1]
        test_probabilities = generate_interpolation_lambda_probabilities(word1,word2)
        for i in range(len(test_probabilities)):
            perplexities[i] = perplexities[i] + math.log(test_probabilities[i],2)
    # apply formula for perplexities
    for i in range(len(perplexities)):
        exponent_formula = -1/len(sequence) * perplexities[i]
        perplexities[i] = 2 ** exponent_formula
    return perplexities


# generates the Pr_Interpolated(word2 | word1) for each lambda : [0.1, 0.3, 0.5, 0.7, 0.9]
# We will return the array of probabilities of (word2|word1) on each lambda for now.
# We will use the output to calculate optimal lambda values later.
def generate_interpolation_lambda_probabilities(word1,word2):
    prob_mle = probability_mle(word1,word2)
    prob_laplace_unigram = probability_laplace_unigram(word2)
    interpolation_prob_array = []
    for i in range(1,10,2):
        lambda_factor = i / 10    # test for lambda = 0.1,0.3,0.5,0.7,0.9
        prob_interpolation = (lambda_factor * prob_mle) + ((1-lambda_factor) * prob_laplace_unigram)
        interpolation_prob_array.append(prob_interpolation)
    return interpolation_prob_array

# After generating the lambda_probabilities array, the caller of the function will choose
# which lambda probability to return, that will give the lowest perplexity (maximum probability).
# lambda_index: {0 : lambda=0.1 },{ 1 : lambda = 0.3} ,{ 2 : lambda=0.5}, {3 : lambda = 0.7}, {4 : lambda = 0.9}
def probability_interpolation(word1,word2,lambda_index):
    lambda_probabilities = generate_interpolation_lambda_probabilities(word1,word2)
    return lambda_probabilities[lambda_index]

#####################
#       SCRIPT      #
#####################
# OPEN FILE
try:
    script, file_name = argv
except ValueError:
    print("Error: the training file .txt should be passed as a parameter\n")
    exit()
f = open(file_name,"r")
text_corpus = f.read()
f.close()

# call functions to build data
build_sentences_array(text_corpus, sentences_array)
# use list of sentences to build dictionaries
build_bigram_tuple_dict(sentences_array,bigram_tuple_dict)
build_unigram_dict(sentences_array,unigram_dict)

try:
    dev_set_corpus = open("dev.txt","r").read()
except FileNotFoundError:
    print("Could not find dev.txt in this directory")
    exit()

# calculate the perplexities. perplexities = array[0:4] where each index is corresponding to lambda 0.1 - 0.9, interval step = 0.2
perplexities = calculate_perplexity(dev_set_corpus)
# find which index of the lambda_probabilities will give the lowest perplexity value.
index_of_min_lambda = perplexities.index(min(perplexities))

# BUILD THE BIGRAM.LM FILE
bigram_file = open("bigram.lm",'w')
for bigram in bigram_tuple_dict.keys():
    first_word = bigram[0]
    second_word = bigram[1]
    bigram_count = bigram_tuple_dict[bigram]
    # use functions to calculate probabilities to write to file.
    prob_mle = probability_mle(first_word,second_word)
    # Calculate PrL(y|x)
    prob_laplace_bigram = probability_laplace_bigram(first_word,second_word)
    # Calculate PrL(x)
    prob_laplace_unigram = probability_laplace_unigram(first_word)
    # PrL(x,y) = PrL(x) * PrL(y|x)
    joint_laplace_prob = prob_laplace_unigram * prob_laplace_bigram
    # store PrL(x,y), to display as top 20 bigrams later.
    joint_prob_array.append((bigram, joint_laplace_prob))
    # PrInt(y|x) = λPrMLE(y|x) + (1−λ)PrL(y).
    # index_of_min_lambda is the index containing the probability calculation using lambda that minimizes perplexity
    prob_interp = probability_interpolation(first_word,second_word,index_of_min_lambda)
    # PrAD =  count(x,y) − D / #(x) where D = 0.5 
    prob_ad = probability_ad(first_word,second_word)
    # write to file, separating each column by tabs
    line = str(first_word)+"\t"+str(second_word)+"\t"+str(bigram_count)+"\t"+str(prob_mle)+"\t"+ str(prob_laplace_bigram)+"\t"+str(prob_interp)+"\t"+str(prob_ad)+"\n"
    bigram_file.write(line)
bigram_file.close()

#BUILD THE UNIGRAM.LM FILE
unigram_file = open("unigram.lm","w")
for unigram in unigram_dict.keys():
    count = unigram_dict[unigram]
    line = str(unigram) + "\t" + str(count) + "\n"
    unigram_file.write(line)
unigram_file.close()

# BUILD THE TOP-BIGRAMS.TXT FILE
top_bigrams_file = open("top-bigrams.txt","w")
joint_prob_array.sort(key=operator.itemgetter(1))
joint_prob_array.reverse()

# GET THE TOP 20 BIGRAMS
top_bigram_tuples = []
result_count = 0
for i in range(len(joint_prob_array)):
    if result_count > 19 :
        break
    first_word = joint_prob_array[i][0][0]
    second_word = joint_prob_array[i][0][1]
    if(first_word =="<s>" or first_word == "</s>" or second_word == "<s>" or second_word == "</s>"):
        continue
    top_bigram_tuples.append(joint_prob_array[i])
    result_count = result_count + 1

# WRITE THE TOP 20 RESULTS TO FILE
for tup in top_bigram_tuples:
    first_word = tup[0][0]
    second_word = tup[0][1]
    joint_prob = str(tup[1])
    top_bigrams_file.write(first_word + "\t" + second_word + "\t" + joint_prob + "\n")
top_bigrams_file.close()
