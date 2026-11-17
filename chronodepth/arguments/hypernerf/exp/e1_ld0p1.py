_base_ = '../default.py'
# E1 stage 1: lambda_depth sweep, l1_aligned, constant schedule, omega 0.9
ModelParams = dict(val_holdout=5)
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', omega_thresh=0.9)
