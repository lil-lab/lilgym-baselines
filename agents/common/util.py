import torch
import torch.optim as optim

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)


def get_optimizer_from_name(optimizer_name, actor_critic, lr, eps, optimizer_sd=None):
    optimizer = None
    if optimizer_name == "adam":
        optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)
    elif optimizer_name == "adamw":
        # parameters from ViLT: Vision-and-Language Transformer Without Convolution or Region Supervision, Kim, Son et al., 2021
        optimizer = optim.AdamW(actor_critic.parameters(), lr=lr, eps=1e-8, betas=(0.9, 0.98))
    
    if optimizer and optimizer_sd:
        optimizer.load_state_dict(torch.load(optimizer_sd))

    if not optimizer:
        raise Exception(f"{optimizer_name} invalid optimizer")
    return optimizer


def get_scheduler_from_name(scheduler_name, optimizer, num_warmup_updates, total_num_updates, scheduler_sd=None):
    scheduler = None
    if scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_updates,
            num_training_steps=total_num_updates,
        )
    elif scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_updates,
            num_training_steps=total_num_updates,
        )
    
    if scheduler and scheduler_sd:
        scheduler.load_state_dict(torch.load(scheduler_sd))
    
    if not scheduler:
        return None
    return scheduler
