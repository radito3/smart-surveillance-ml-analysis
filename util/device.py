import torch


def get_device() -> torch.device:
    if torch.accelerator.is_available():
         return torch.accelerator.current_accelerator()
    else:
        return torch.device('cpu')
