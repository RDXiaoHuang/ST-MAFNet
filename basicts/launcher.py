from typing import Dict, Union

import easytorch

def launch_training(cfg: Union[Dict, str], gpus: str = None, node_rank: int = 0):
    """Extended easytorch launch_training.

    Args:
        cfg (Union[Dict, str]): Easytorch config.
        gpus (str): set ``CUDA_VISIBLE_DEVICES`` environment variable.
        node_rank (int): Rank of the current node.
    """

    # pre-processing of some possible future features, such as:
    # registering model, runners.
    # config checking
    pass
    # launch training based on easytorch
    # Try different parameter names for different easytorch versions
    try:
        # Try gpus parameter first (for most easytorch versions)
        easytorch.launch_training(cfg=cfg, gpus=gpus, node_rank=node_rank)
    except TypeError as e1:
        if "unexpected keyword argument" in str(e1):
            try:
                # Try devices parameter (for some easytorch versions)
                easytorch.launch_training(cfg=cfg, devices=gpus, node_rank=node_rank)
            except TypeError as e2:
                # Fallback: try without explicit gpus parameter
                try:
                    easytorch.launch_training(cfg=cfg, node_rank=node_rank)
                except Exception:
                    # If all fail, raise the original error
                    raise e1
        else:
            raise e1

