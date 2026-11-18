_base_ = '../coffee_martini.py'
# B1: lambda_d = 1.0 (centre 0.1), full setting
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=1.0, lambda_depth_schedule='const', lambda_temp=0.1,
                   omega_thresh=0.9, align='per_camera', depth_prior_space='disparity')
