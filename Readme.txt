#########################################
            INSTRUCTIONS:
#########################################
1) Install python 3, and make sure it is on environment path. All development is tested against python v3.5.3
2) Open cmd on windows, or linux terminal on the dir of the project.
3) If both python2 and python3 is installed, you can explicitly run python 3 in substitute of all commands using:
      py  -3  exampleProgramFile.py  extraInputArguments
3) To generate language model files (.lm) run:
    python LanguageModelBuilder.py train.txt
4) top-bigrams.txt, bigram.lm and unigram.lm should be generated in the same directory.
5) To query the joint conditional probabilities generated from statistics in the language model files
    a) MLE Probabilities
          python bigram-query.py bigram.lm unigram.lm word1 word2 M
    b) Laplace Probabilities
          python bigram-query.py bigram.lm unigram.lm word1 word2 L
    c) Interpolated Probabilities
          python bigram-query.py bigram.lm unigram.lm word1 word2 I
    d) Katz-Backoff Probabilities / AD probabilities(if word1 and word2 were seen in training.txt)
          python bigram-query.py bigram.lm unigram.lm word1 word2 K
6) To evaluate the perplexity of the language models run:
          python perplexity.py bigram.lm unigram.lm test.txt

############################################
        RESULTS
###########################################
After running the language model builder and evaluating the language models, you should see the following perplexities:
  Laplace Bigram Perplexity : 1409.4357121475173
  Interpolated Bigram Perplexity : 475.2388914547642
  Laplace Unigram Perplexity : 533.0164865169504

Top 20 bigrams based on joint laplace probability , PrL(x,y)
After removing values resulting from starting/ending tokens
---------------------------------------------------------------
of	the	0.0010642905898687355
in	the	0.0008247583709953834
the	fly	0.00038167280505791936
the	child	0.0002694160976879431
the	body	0.0002694160976879431
the	house	0.00020206207326595732
the	most	0.00020206207326595732
of	insects	0.00018953120093552822
and	the	0.00018624163803587998
to	the	0.0001834453937743572
the	wings	0.00017961073179196207
the	home	0.00017961073179196207
of	a	0.00017495187778664143
of	these	0.00016037255463775463
the	rural	0.0001571593903179668
the	mouth	0.0001571593903179668
the	mosquito	0.00013470804884397155
the	maggots	0.00013470804884397155
the	fall	0.00013470804884397155
the	bottom	0.00013470804884397155

The smallest perplexity resulted from the interpolated bigram perplexity ~ 475,
followed by laplace unigram perplexity ~ 533, and finally by laplace bigram perplexity ~1409.

The results mean that the interpolated method worked the best.
Possible reasons for this may be because some of the MLE  probabilities from the training set was useful in modeling the test set,
but unseen words in the test set occurred enough to throw the perplexity in favor of the interpolation method.
This way, it was an average method, between the laplace unigram probability and the MLE probability.

This is supported by the fact that the laplace unigram perplexity also did decently well, at ~533.
Which means that certain words' frequencies, as seen in the training data, was a good predictor for the test set.
The fact that laplace bigram perplexity did the worst, can mean that the conditional probability of one word following another,
was not very strong as seen in the test set.
