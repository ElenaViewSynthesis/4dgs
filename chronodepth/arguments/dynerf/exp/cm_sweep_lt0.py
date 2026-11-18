_base_ = '../coffee_martini.py'
# D1: lambda_t = 0, depth term only (centre 0.1), full setting
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', lambda_temp=0.0,
                   omega_thresh=0.9, align='per_camera', depth_prior_space='disparity')
