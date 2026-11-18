_base_ = '../coffee_martini.py'
# C: Omega threshold 0.99 (centre 0.9), full setting
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', lambda_temp=0.1,
                   omega_thresh=0.99, align='per_camera', depth_prior_space='disparity')
