import torch

from helper_functions.general_helper_functions import add_weight_decay

def create_optimizer(model, args):
    # parameters = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(params=parameters, lr=args.lr, weight_decay=args.weight_decay)  # true wd, filter_bias_and_bn
    return optimizer

def create_optimizer_adam(model, args):
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9,0.999))  # true wd, filter_bias_and_bn
    return optimizer

def create_optimizer_sgd(model, args):
    parameters = add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.SGD(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    return optimizer

def update_optimizer(model, learning_rate, args):
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9,0.999)
    )

    return optimizer
