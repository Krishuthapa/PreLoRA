from helper_functions.distributed import print_at_master

def add_weight_decay(model, weight_decay=1e-4, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy2(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""

    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)                      
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    return [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]

def silence_PIL_warnings():
    import PIL
    wa = PIL.Image.warnings
    wa.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

class HookController:
    def __init__(self, interval=10):
        self.interval = interval
        self.iteration = 0

    def increment(self):
        self.iteration += 1

    def should_run(self):
        return (self.iteration % self.interval) == 0

def check_container_and_assign(container, key, default_value = None):
    if key in container:
        return container[key]
    
    return default_value