_base_ = '../coffee_martini.py'
# Block A, sparse 5 + depth: five cameras left out of the training set
ModelParams = dict(exclude_cams='cam04,cam09,cam10,cam13,cam20')
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', lambda_temp=0.1,
                   omega_thresh=0.9, align='per_camera', depth_prior_space='disparity')
