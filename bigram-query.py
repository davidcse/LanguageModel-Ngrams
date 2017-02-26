from sys import argv

#### STORAGE DICTIONARIES #####
bigram_data_dict={}
unigram_data_dict={}
subsequent_words_bank= {}

#### GRAB THE TRAINED DATA FROM THE FILES
def open_retrieve_probability_data(lm_file):
    f = open(lm_file.strip(),"r")
    probability_data = f.read().split("\n")
    f.close()
    return probability_data

##### takes bigram and unigram probability data, and inserts it into bigram_data_dict and unigram_data_dict
def parse_probability_data_into_maps(bigram_prob_data,unigram_prob_data, bigram_data_dict, unigram_data_dict):
    # BUILD BIGRAM DICTIONARY FROM LINES OF PROBABILITY DATA
    for i in range(len(bigram_prob_data)):
        line = bigram_prob_data[i].split()
        if(len(line) == 0):
            continue
        first_word = line[0]
        second_word = line[1]
        bigram_data_dict[(first_word,second_word)] = line
        if first_word in subsequent_words_bank:
            subsequent_words_list = subsequent_words_bank[first_word]
            subsequent_words_list.append(second_word)
        else:
            subsequent_words_bank[first_word] = [second_word]
    # BUILD UNIGRAM DICTIONARY FROM LINES OF PROBABILITY DATA
    for i in range(len(unigram_prob_data)):
        line = unigram_prob_data[i].split()
        if(len(line) == 0):
            continue
        unigram_data_dict[line[0]] = line


# Pr_L(y)
def laplace_single_word(word):
    global total_num_words
    global vocabulary_size
    count_word = 0 if word not in unigram_data_dict else int(unigram_data_dict[word][1])
    laplace_prob = (count_word + 1) / (total_num_words + vocabulary_size + 1)
    return laplace_prob

# alpha(x)
def katz_backoff_alpha(word1):
    summation_ad_prob = 0
    if word1 in subsequent_words_bank:
        subsequent_words_list = subsequent_words_bank[word1]
    else:
        subsequent_words_list=[]
    for subseq_word in subsequent_words_list:
        tup = (word1,subseq_word)
        summation_ad_prob = summation_ad_prob + float(bigram_data_dict[tup][6])
    alpha_prob = 1 - summation_ad_prob
    return alpha_prob

# beta(y)
def katz_backoff_beta(word1,word2):
    summation_laplace_nonsubsequent_words = 0
    if word1 in subsequent_words_bank:
        subsequent_words_list = subsequent_words_bank[word1]
    else:
        subsequent_words_list = []
    for word in unigram_data_dict.keys():
        if word not in subsequent_words_list:
            laplace_prob_word = laplace_single_word(word)
            summation_laplace_nonsubsequent_words = summation_laplace_nonsubsequent_words + laplace_prob_word
    beta_prob = laplace_single_word(word2) / summation_laplace_nonsubsequent_words
    return beta_prob

# alpha(x)* beta(y)
def katz_backoff_prob(word1,word2):
    alpha_result = katz_backoff_alpha(word1)
    beta_result = katz_backoff_beta(word1,word2)
    katz_backoff =  alpha_result * beta_result
    return katz_backoff


#########SCRIPT #################
try:
    script, bigram_file, unigram_file, search_word1, search_word2, smoothing_option  = argv
except ValueError:
    print("Please enter --bigram file path  --unigram file path --search word 1 -- search word2 --smoothing option")
    exit()

# M,L,I,K options
smoothing_option = smoothing_option.upper().strip()
# words x,y
search_word1 = search_word1.strip()
search_word2 = search_word2.strip()
# build dictionaries from lines of probability data retrieved from .lm files.
probability_data_bigrams = open_retrieve_probability_data(bigram_file)
probability_data_unigrams = open_retrieve_probability_data(unigram_file)
parse_probability_data_into_maps(probability_data_bigrams, probability_data_unigrams ,bigram_data_dict,unigram_data_dict)

#### PROCESS USER SMOOTHING CHOICE #######
probability_type = "Smoothing Method Not Specified"
column_index = 0
if(smoothing_option == "M"):
    column_index = 3
    probability_type = "MLE probabilities"
elif(smoothing_option == "L"):
    column_index = 4
    probability_type = "Laplace probabilities"
elif(smoothing_option == "I"):
    column_index = 5
    probability_type = "Interpolation probabilities"
elif(smoothing_option == "K"):
    column_index = 6
    probability_type = "Katz-Backoff(AD) probabilities"


######### CALCULATION VARIABLES
tup = (search_word1,search_word2)
vocabulary_size = len(list(unigram_data_dict.keys()))
lambda_fact = 0.1
total_num_words = 0
for unigram_lines in unigram_data_dict.values():
    total_num_words = total_num_words + int(unigram_lines[1])


######### BEGIN PRINTING TO USER STDOUT
print(probability_type + ":\n")
## use precalculated probability from lookup in map
if tup in bigram_data_dict:
    print(bigram_data_dict[tup][column_index])
# IF NOT SEEN BEFORE (word1,word2)
elif column_index ==3 :
    print("dynamically calculated: ")
    print(0)
elif column_index == 4:
    # using formula:  count(x,y) + 1 / (count(x) + vocabulary_size + 1)
    count_word1 = 0 if search_word1 not in unigram_data_dict else int(unigram_data_dict[search_word1][1])
    unseen_laplace_bigram_prob = 1 / (count_word1 + vocabulary_size + 1)
    print("dynamically calculated: ")
    print(unseen_laplace_bigram_prob)
elif column_index == 5:
    # using formula:  lambda*Pr_mle(y|x) + (1-lambda)*Pr_laplace(y). Note:Pr_mle(y|x) = 0
    laplace_prob_word2 = laplace_single_word(search_word2)
    unseen_interpolation_prob = ((1-lambda_fact) * laplace_prob_word2)
    print("dynamically calculated: ")
    print(unseen_interpolation_prob)
elif column_index ==6:
    katz_backoff = katz_backoff_prob(search_word1,search_word2)
    print("dynamically calculated: ")
    print(katz_backoff)
