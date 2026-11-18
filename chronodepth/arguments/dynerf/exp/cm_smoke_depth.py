_base_ = '../coffee_martini.py'
# Smoke test of the multi-camera depth code only: 300 coarse + 600 fine iterations, not an experiment
OptimizationParams = dict(iterations=600, coarse_iterations=300)
ModelParams = dict(exclude_cams='cam11,cam19')
DepthParams = dict(depth_loss='l1_aligned', lambda_depth=0.1, lambda_depth_schedule='const', lambda_temp=0.1,
                   omega_thresh=0.9, align='per_camera', depth_prior_space='depth')
