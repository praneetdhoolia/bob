"""Exact optimizer/random continuation, separate from weights-only initialization."""
import random
import torch


def capture_training(optimizer, names, configuration, device):
    return dict(optimizerState=optimizer.state_dict(),optimizerParameterNames=names,
        trainingConfiguration=configuration,pythonRandomState=random.getstate(),
        torchRandomState=torch.get_rng_state(),
        cudaRandomState=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)


def restore_training(checkpoint, optimizer, names, configuration, device):
    required=['optimizerState','pythonRandomState','torchRandomState']
    if (checkpoint.get('trainingConfiguration')!=configuration
        or checkpoint.get('optimizerParameterNames')!=names
        or any(key not in checkpoint for key in required)):
        return False
    optimizer.load_state_dict(checkpoint['optimizerState'])
    random.setstate(checkpoint['pythonRandomState'])
    torch.set_rng_state(checkpoint['torchRandomState'].cpu())
    if device.startswith('cuda') and checkpoint.get('cudaRandomState') is not None:
        torch.cuda.set_rng_state_all(checkpoint['cudaRandomState'])
    return True
