_base_ = '../coffee_martini.py'
# Block A, sparse 2 + depth: cam11 and cam19 left out of the training set
ModelParams = dict(exclude_cams='cam11,cam19')
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', lambda_temp=0.1,
                   omega_thresh=0.9, align='per_camera', depth_prior_space='disparity')
