from torch.optim.lr_scheduler import LambdaLR

def build_lr_scheduler(cfg, optimizer):
    solver = cfg['SOLVER']
    max_iter      = solver.get('MAX_ITER', 1000)
    steps         = solver.get('STEPS', [])
    gamma         = solver.get('GAMMA', 0.1)
    warmup_iters  = solver.get('WARMUP_ITERS', 10)
    warmup_factor = solver.get('WARMUP_FACTOR', 1.0)

    def lr_lambda(iteration):
        if iteration < warmup_iters:
            alpha = iteration / max(1, warmup_iters)
            return warmup_factor * (1 - alpha) + alpha
        factor = 1.0
        for step in steps:
            if iteration >= step:
                factor *= gamma
        return factor

    return LambdaLR(optimizer, lr_lambda=lr_lambda)
