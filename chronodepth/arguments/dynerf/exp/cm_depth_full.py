_base_ = '../coffee_martini.py'
# Block A, full + depth: all 17 training cameras; prior aligned in inverse depth (prior-space check on this scene)
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', lambda_temp=0.1,
                   omega_thresh=0.9, align='per_camera', depth_prior_space='disparity')
