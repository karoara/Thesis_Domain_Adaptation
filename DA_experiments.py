# DA EXPERIMENT SETUP ---------------------------------------------------------
# 
# Setup for different domain adaptation experiments (conducted by DA_test,
# DA_test_lang scripts).


# AMAZON DA TESTS -------------------------------------------------------------
# Domain numbers correspond to different domains from the amazon data.

amazon_experiments = {} # test_i --> [[training domains], [testing domains]]

set1 = [[i for i in range(6)], [i for i in range(6, 12)],
        [i for i in range(12, 17)], [i for i in range(17, 22)]]

amazon_experiments["e1"] = [set1[0] + set1[1] + set1[2], set1[3]]
amazon_experiments["e2"] = [set1[0] + set1[1] + set1[3], set1[2]]
amazon_experiments["e3"] = [set1[0] + set1[2] + set1[3], set1[1]]
amazon_experiments["e4"] = [set1[1] + set1[2] + set1[3], set1[0]]


# REDDIT DA TESTS -------------------------------------------------------------
# Domain numbers correspond to different domains from the reddit data.

reddit_experiments = {}

reddit_experiments["e1"] = [[0, 1, 2, 4, 5, 6, 8, 11, 12, 14, 15, 16, 18, 19, 21, 22, 23, 25, 26, 29], \
        [3, 7, 9, 10, 13, 17, 20, 24, 27, 28, 30]]
reddit_experiments["e2"] = [[0, 1, 2, 3, 7, 9, 10, 12, 13, 14, 17, 18, 20, 22, 24, 27, 28, 30], \
        [4, 5, 6, 8, 11, 15, 16, 19, 21, 23, 25, 26, 29]]


# SYNTHETIC (1) DA TESTS ------------------------------------------------------
# Domain numbers correspond to different domains from the synth1 data.

synth1_experiments = {}

synth1_experiments["e1"] = [[0, 2, 10, 14, 15], [1, 7, 13, 16, 19], [8, 17], [4, 11]]
synth1_experiments["e2"] = [[0, 3, 10, 14, 17], [1, 4, 7, 11, 16], [15, 18], [13, 19]]


# SYNTHETIC (2) DA TESTS ------------------------------------------------------
# Domains correspond to the domains from the synth2 data.

synth2_experiments = {}

synth2_experiments["e1"] = [[0, 1]]

# SYNTHETIC (3) DA TESTS ------------------------------------------------------

synth3_experiments = {} # test_i --> [# datapoints per domain, # training domains, # test]

synth3_experiments["e1"] = [1000, 20, 5]
synth3_experiments["e2"] = [1000, 60, 10]
synth3_experiments["e3"] = [40, 1500, 300]

