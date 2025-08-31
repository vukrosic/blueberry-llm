"""
Optimizer implementations including Muon
"""
import torch
import torch.nn as nn
from typing import List, Tuple
from configs.base_config import ExperimentConfig

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()

    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT

    return X

class Muon(torch.optim.Optimizer):
    """Muon - MomentUm Orthogonalized by Newton-schulz"""
    
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g.view_as(p), alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

def create_optimizer(model: nn.Module, config: ExperimentConfig) -> Tuple[List[torch.optim.Optimizer], List[str]]:
    """Create optimizers with hybrid approach (Muon + AdamW)"""
    muon_params = []
    adamw_params = []
    
    # Separate parameters for different optimizers
    for name, param in model.named_parameters():
        if (param.ndim == 2 and 
            'token_embedding' not in name and 
            'norm' not in name and 
            param.requires_grad):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    
    optimizers = []
    optimizer_names = []
    
    # Muon for 2D parameters (weights)
    if muon_params:
        muon_optimizer = Muon(muon_params, lr=config.learning_rate, momentum=0.95)
        optimizers.append(muon_optimizer)
        optimizer_names.append("Muon")
    
    # AdamW for other parameters (embeddings, norms, biases)
    if adamw_params:
        adamw_optimizer = torch.optim.AdamW(
            adamw_params, 
            lr=config.learning_rate * 0.1,  # Lower LR for AdamW
            weight_decay=config.weight_decay
        )
        optimizers.append(adamw_optimizer)
        optimizer_names.append("AdamW")
    
    return optimizers, optimizer_names

def create_schedulers(optimizers: List[torch.optim.Optimizer], config: ExperimentConfig) -> List[torch.optim.lr_scheduler.LRScheduler]:
    """Create learning rate schedulers"""
    schedulers = []
    
    for optimizer in optimizers:
        warmup_steps = config.max_steps // 20
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(3.14159 * progress)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)
    
    return schedulers