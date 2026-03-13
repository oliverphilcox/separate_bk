import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, dim, resblock_dim=None):
        super().__init__()
        if resblock_dim is None:
            resblock_dim = dim

        self.net = nn.Sequential(
            nn.Linear(dim, resblock_dim),
            nn.ReLU(),
            nn.Linear(resblock_dim, dim),
        )

        # Layer normalization for stability
        self.norm = nn.LayerNorm(dim)
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.norm(x + self.net(x)))

class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        self.output_proj = nn.Linear(hidden_dim, 1)

        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.output_proj(x)

class SeparableApproximation(nn.Module):
    def __init__(self, num_terms=2, N_models=1, symm_kind=1, sub_arch='ResMLP', log_transform=False, add_bias=False, kpivot=0.05, optimal_weights=False):
        super().__init__()
        self.num_terms = num_terms
        self.N_models = N_models
        self.log_transform = log_transform
        self.sub_arch = sub_arch
        self.kpivot = kpivot
        self.optimal_weights = optimal_weights
        if sub_arch=="MLP":
            self.alpha = nn.ModuleList([MLP(1, hidden_dim=64) for _ in range(num_terms)])
            if symm_kind !=3:
                self.beta = nn.ModuleList([MLP(1, hidden_dim=64) for _ in range(num_terms)])
            if symm_kind == 1:
                self.gamma = nn.ModuleList([MLP(1, hidden_dim=64) for _ in range(num_terms)])
        elif sub_arch=='ResMLP':
            self.alpha = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1) for _ in range(num_terms)])
            if symm_kind !=3:
                self.beta = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1) for _ in range(num_terms)])
            if symm_kind == 1:
                self.gamma = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1) for _ in range(num_terms)])
        else:
            raise ValueError('architecture of submodules not defined')
        if not optimal_weights:
            self.weights = nn.Parameter(torch.ones(num_terms+add_bias,N_models))

        self.symm_kind = symm_kind
        self.add_bias = add_bias

    @classmethod
    def copy_model(cls, old_model, new_num_terms=0):
        """Returns a new model instance, copying existing parameters for the old terms and randomly initializing the new ones."""
        new_model = cls(
            num_terms=new_num_terms,
            N_models=old_model.N_models,
            symm_kind=old_model.symm_kind,
            sub_arch=old_model.sub_arch,
            log_transform=old_model.log_transform,
            add_bias=old_model.add_bias,
            kpivot=old_model.kpivot,
            optimal_weights=old_model.optimal_weights
        )

        # Copy previous alpha, beta, gamma, weight functions
        for i in range(old_model.num_terms):
            new_model.alpha[i].load_state_dict(old_model.alpha[i].state_dict())
            if hasattr(old_model, 'beta'):
                new_model.beta[i].load_state_dict(old_model.beta[i].state_dict())
            if hasattr(old_model, 'gamma'):
                new_model.gamma[i].load_state_dict(old_model.gamma[i].state_dict())
        with torch.no_grad():
            if not old_model.optimal_weights:
                new_model.weights[:, :] = 0
                new_model.weights[:old_model.num_terms, :] = old_model.weights

        return new_model

    def encoder(self, x):
        """Encode k to log(k/k_pivot)"""
        return torch.log(x/self.kpivot)

    def _apply_transform(self, x):
        """Apply log transform if set."""
        if self.log_transform:
            return torch.exp(x)
        return x

    def _compute_term(self, i, k1, k2, k3):
        """Compute the symmetrized basis term for a given index."""
        f = self._apply_transform

        if self.symm_kind==0:
            assert self.log_transform, "Need log-transform to use symm_kind=0"
            a1, a2, a3 = f(self.alpha[i](k1)), f(self.alpha[i](k2)), f(self.alpha[i](k3))
            b1, b2, b3 = f(self.beta[i](k1)), f(self.beta[i](k2)), f(self.beta[i](k3))
            return (a1*b2/(a3*b3) + a2*b3/(a1*b1) + a3*b1/(a2*b2) + a2*b1/(a3*b3) + a3*b2/(a1*b1) + a1*b3/(a2*b2))/6.

        elif self.symm_kind==1:
            a1, a2, a3 = f(self.alpha[i](k1)), f(self.alpha[i](k2)), f(self.alpha[i](k3))
            b1, b2, b3 = f(self.beta[i](k1)), f(self.beta[i](k2)), f(self.beta[i](k3))
            c1, c2, c3 = f(self.gamma[i](k1)), f(self.gamma[i](k2)), f(self.gamma[i](k3))
            return (a1*b2*c3+a1*b3*c2+a2*b1*c3+a2*b3*c1+a3*b1*c2+a3*b2*c1)/6.

        elif self.symm_kind==2:
            a1, a2, a3 = f(self.alpha[i](k1)), f(self.alpha[i](k2)), f(self.alpha[i](k3))
            b1, b2, b3 = f(self.beta[i](k1)), f(self.beta[i](k2)), f(self.beta[i](k3))
            return (a1*b2*b3+a2*b1*b3+a3*b1*b2)/3.

        elif self.symm_kind==3:
            a1, a2, a3 = f(self.alpha[i](k1)), f(self.alpha[i](k2)), f(self.alpha[i](k3))
            return a1*a2*a3

    def get_predictions(self, x):
        """Compute the basis terms (without weights) for all input triangles. Returns shape (num_terms, N)."""
        k1 = self.encoder(x[:,0]).view(-1,1)
        k2 = self.encoder(x[:,1]).view(-1,1)
        k3 = self.encoder(x[:,2]).view(-1,1)

        terms = torch.zeros((self.num_terms, len(x)), device=x.device)
        for i in range(self.num_terms):
            terms[i] = self._compute_term(i, k1, k2, k3).ravel()

        assert not self.add_bias
        return terms

    def forward(self, x):
        if self.optimal_weights:
            raise Exception("Use get_predictions() with optimal_weights=True")

        k1 = self.encoder(x[:,0]).view(-1,1)
        k2 = self.encoder(x[:,1]).view(-1,1)
        k3 = self.encoder(x[:,2]).view(-1,1)

        output = 0
        for i in range(self.num_terms):
            output += self.weights[i]*self._compute_term(i, k1, k2, k3)

        if self.add_bias:
            output += self.weights[-1]
        return output
