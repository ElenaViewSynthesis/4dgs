_base_ = '../coffee_martini.py'
# Block A, sparse 2 baseline: cam11 and cam19 left out of the training set
ModelParams = dict(exclude_cams='cam11,cam19')
