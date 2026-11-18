_base_ = '../coffee_martini.py'
# Block A, sparse 5 baseline: five cameras left out of the training set
ModelParams = dict(exclude_cams='cam04,cam09,cam10,cam13,cam20')
