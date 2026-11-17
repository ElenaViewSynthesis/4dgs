_base_ = '../default.py'
# E1: baseline on the reduced training split (every 5th training frame held out)
ModelParams = dict(val_holdout=5)
