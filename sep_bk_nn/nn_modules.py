import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            #nn.ReLU(), 
            nn.Tanh(),
            #nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            #nn.ReLU(),
            nn.Tanh(),#?
            #nn.GELU(),
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
        # Residual connection
        return self.act(self.norm(x + self.net(x)))
    
class ResMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_blocks=3):
        super().__init__()
        
        # Initial projection to hidden dimension
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(hidden_dim) for _ in range(num_blocks)
        ])
        
        # Final projection to output
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        # Initialize weights using Kaiming initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Initial projection
        x = self.input_proj(x)
        
        # Residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Final projection
        return self.output_proj(x)

class SeparableApproximation(nn.Module):
    def __init__(self, num_terms=2, N_models=1, symm_kind=1, sub_arch='ResMLP', log_transform=False, add_bias=False, kpivot=0.05):
        super().__init__()
        self.num_terms = num_terms
        self.N_models = N_models
        self.log_transform = log_transform
        self.sub_arch = sub_arch
        self.kpivot = kpivot
        if sub_arch=="MLP":
            self.alpha = nn.ModuleList([MLP(1, hidden_dim=64) for _ in range(num_terms)])
            if symm_kind !=3:
                self.beta = nn.ModuleList([MLP(1, hidden_dim=64) for _ in range(num_terms)])
            if symm_kind == 1:
                self.gamma = nn.ModuleList([MLP(1, hidden_dim=64) for _ in range(num_terms)])
        elif sub_arch=='ResMLP':
            self.alpha = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1) for _ in range(num_terms)])
            if symm_kind !=3:
                self.beta = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1)  for _ in range(num_terms)])
            if symm_kind == 1:
                self.gamma = nn.ModuleList([ResMLP(1,hidden_dim=64, num_blocks=1)  for _ in range(num_terms)])
        else:
            raise ValueError('architecture of submodules not defined')
        self.weights = nn.Parameter(torch.ones(num_terms+add_bias,N_models))

        # Define symmetry
        self.symm_kind = symm_kind
        self.add_bias = add_bias

    @classmethod
    def copy_model(cls, old_model, new_num_terms=0):
        """
        Returns a new model instance, copying existing parameters for the old terms and randomly initializing the new ones.
        """
        new_model = cls(
            num_terms=new_num_terms,
            N_models=old_model.N_models,
            symm_kind=old_model.symm_kind,
            sub_arch=old_model.sub_arch,
            log_transform=old_model.log_transform,
            add_bias=old_model.add_bias,
            kpivot=old_model.kpivot
        )

        # Copy previous alpha, beta, gamma, weight functions
        for i in range(old_model.num_terms):
            new_model.alpha[i].load_state_dict(old_model.alpha[i].state_dict())
            if hasattr(old_model, 'beta'):
                new_model.beta[i].load_state_dict(old_model.beta[i].state_dict())
            if hasattr(old_model, 'gamma'):
                new_model.gamma[i].load_state_dict(old_model.gamma[i].state_dict())
        with torch.no_grad():
            new_model.weights[:, :] = 0
            new_model.weights[:old_model.num_terms, :] = old_model.weights
            
        return new_model

    def encoder(self, x):
        """Encode k to log(k/k_pivot)"""
        return torch.log(x/self.kpivot)
    
    def forward(self, x):
        if self.log_transform:
            f = lambda x: torch.exp(x)
        else:
            f = lambda x: x

        # Optionally order weights
        #sorted_weights, indices = torch.sort(self.weights[:self.num_terms], descending=True)
        sorted_weights, indices = self.weights, torch.arange(self.num_terms)
        
        # Define ks, redefining k -> log(k / k_pivot). 
        k1 = self.encoder(x[:,0]).view(-1,1)
        k2 = self.encoder(x[:,1]).view(-1,1)
        k3 = self.encoder(x[:,2]).view(-1,1)
        if not hasattr(self, 'pivot_scale'):
            self.pivot_scale = self.encoder(x[0,0].view(1)*0+self.kpivot)
        
        def get_abc(i,k1,k2,k3):
            # Symmetry kind 0: enforcing full symmetry and full scale invariance, 2 functions alpha and beta, with 6 permutations of alpha_1/alpha_3*beta_2/beta_3
            if self.symm_kind==0:
                assert self.log_transform, "Need log-transform to use symm_kind=0"
                a1 = f(self.alpha[i](k1))
                a2 = f(self.alpha[i](k2))
                a3 = f(self.alpha[i](k3))
                b1 = f(self.beta[i](k1))
                b2 = f(self.beta[i](k2))
                b3 = f(self.beta[i](k3))
                return (a1*b2/(a3*b3) + a2*b3/(a1*b1) + a3*b1/(a2*b2) + a2*b1/(a3*b3) + a3*b2/(a1*b1) + a1*b3/(a2*b2))/6.
            
            # Symmetry kind 1: enforcing full symmetry, 3 different functions with 6 permutations
            elif self.symm_kind==1:
                a1 = f(self.alpha[i](k1))
                a2 = f(self.alpha[i](k2))
                a3 = f(self.alpha[i](k3))
                b1 = f(self.beta[i](k1))
                b2 = f(self.beta[i](k2))
                b3 = f(self.beta[i](k3))
                c1 = f(self.gamma[i](k1))
                c2 = f(self.gamma[i](k2))
                c3 = f(self.gamma[i](k3))
                #norm = f(self.alpha[i](self.pivot_scale))*f(self.beta[i](self.pivot_scale))*f(self.gamma[i](self.pivot_scale))
                return (a1*b2*c3+a1*b3*c2+a2*b1*c3+a2*b3*c1+a3*b1*c2+a3*b2*c1)/6.#/norm
            
            # Symmetry kind 2: assuming additional symmetry, 2 functions alpha and beta, with 3 permutations
            elif self.symm_kind==2:
                a1 = f(self.alpha[i](k1))
                a2 = f(self.alpha[i](k2))
                a3 = f(self.alpha[i](k3))
                b1 = f(self.beta[i](k1))
                b2 = f(self.beta[i](k2))
                b3 = f(self.beta[i](k3))
                #norm = f(self.alpha[i](self.pivot_scale))*f(self.beta[i](self.pivot_scale))**2
                return (a1*b2*b3+a2*b1*b3+a3*b1*b2)/3.#/norm
            
            # Symmetry kind 3: assuming full symmetry, 1 function alpha 
            elif self.symm_kind==3:
                a1 = f(self.alpha[i](k1))
                a2 = f(self.alpha[i](k2))
                a3 = f(self.alpha[i](k3))
                #norm = f(self.alpha[i](self.pivot_scale))
                return a1*a2*a3#/norm

        ## Define outputs
        output = 0
        for i in range(self.num_terms):
            # Compute symmetrized output with weights
            output += sorted_weights[i]*get_abc(indices[i],k1,k2,k3)
        
        # Optionally add a bias term
        if self.add_bias:
            output += self.weights[-1]    
        return output    