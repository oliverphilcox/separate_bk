import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import interpolate, integrate
import os, time
from .nn_modules import SeparableApproximation

class KFilter(torch.nn.Module):
    def __init__(self, filename, log_interp=False):
        super().__init__()
        data = np.load(filename)
        k_vals, f_vals = data[:, 0], data[:, 1]
        
        # Ensure monotonic increasing k
        assert np.all(np.diff(k_vals) > 0), "k-values must be increasing"

        self.register_buffer('k', torch.tensor(k_vals, dtype=torch.float32))
        self.register_buffer('f', torch.tensor(f_vals, dtype=torch.float32))
        self.log_interp = log_interp

    def forward(self, k_input):
        device = k_input.device  # get input device

        # Move buffers to the right device
        k = self.k.to(device)
        f = self.f.to(device)

        x_input = torch.log(k_input) if self.log_interp else k_input
        x = torch.log(k) if self.log_interp else k

        x_input = x_input.clamp(x[0].item(), x[-1].item())

        idx_hi = torch.searchsorted(x, x_input)
        idx_hi = torch.clamp(idx_hi, 1, len(x) - 1)
        idx_lo = idx_hi - 1

        x_lo = x[idx_lo]
        x_hi = x[idx_hi]
        f_lo = f[idx_lo]
        f_hi = f[idx_hi]

        weight_hi = (x_input - x_lo) / (x_hi - x_lo)
        weight_lo = 1 - weight_hi
        f_interp = f_lo * weight_lo + f_hi * weight_hi

        # Cache on same device
        self._cache_key = k_input.clone().detach()
        self._cache_value = f_interp.detach()

        return f_interp

class SepBKNN:
    def __init__(self, num_terms, symm_kind, add_bias=False, loss_func = 'mse', sub_arch='MLP', kpivot=0.05, filterfile = None, log_transform=False, N_models=1, device=None):

        self.device = device
        self.N_models = N_models
        self.model = SeparableApproximation(num_terms=num_terms, N_models=N_models, symm_kind=symm_kind, add_bias=add_bias, sub_arch=sub_arch, log_transform=log_transform, kpivot=kpivot).to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.3)
        self.loss_func = loss_func
        self.num_terms = num_terms
        self.symm_kind = symm_kind
        self.add_bias = add_bias
        self.filterfile = filterfile
        self.kpivot = kpivot
        if loss_func == 'mse':
            self.criterion = nn.MSELoss()
            
        if self.filterfile is not None:
            try:
                self.filter = KFilter(self.filterfile, log_interp=True)
            except IOError:
                raise IOError(f"Error: Filterfile {self.filterfile} not found.")                
            
    def save_checkpoint(self, val_loss, epoch, checkpoint_dir='./models/checkpoint'):
        """Save model checkpoint"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'num_terms': self.num_terms,
            'symm_kind': self.symm_kind,
            'add_bias': self.add_bias
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pth')
        torch.save(checkpoint, checkpoint_path)
        
    def load_checkpoint(self, checkpoint_dir='./models/checkpoint'):
        """Load model checkpoint"""
        checkpoint_path = os.path.join(checkpoint_dir, f'best_model.pth')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint['epoch'], checkpoint['val_loss']
        return None, None

    def k_weighting(self, X, kpower=0):
        """Apply an appropriate k-space weighting filter. We add a factor of 1/(k1+k2+k3) to convert from 2D to 3D (relevant for the CMB). We also add a factor of k1 k2 k3 since we sample k in log-space. Finally, we apply an approximate signal-to-noise based filter."""
        k_weight = (X[:,0]*X[:,1]*X[:,2])**kpower/(X[:,0]+X[:,1]+X[:,2])
        if self.filterfile is not None:
            k_weight *= self.filter(X[:,0])*self.filter(X[:,1])*self.filter(X[:,2])
        return k_weight
        
    def inner_product_loss(self, X, predicted_y, true_y):
        """Compute the inner-product loss."""
        return torch.sum(self.k_weighting(X)[:,None]*(predicted_y - true_y)**2,axis=0)/len(predicted_y)
        
    def cosine_loss(self, X, predicted_y, true_y):
        """Compute the squared cosine loss""
        """
        k_weight = self.k_weighting(X)[:,None]
        innerPP = torch.sum(k_weight*predicted_y**2,axis=0)
        innerTT = torch.sum(k_weight*true_y**2,axis=0)
        innerPT = torch.sum(k_weight*predicted_y*true_y,axis=0)
        return 1.0 - innerPT**2./(innerPP*innerTT)
    
    def train(self, train_dataset, val_dataset, model_nos, epochs, checkpoint_dir, batch_size=512, patience=5):
        train_losses = []
        val_losses = []
        
        best_val_loss = float('inf')
        
        # Initialize early stopping
        early_stopping = EarlyStopping(patience=patience, min_val=1e-4, min_delta=1e-6, verbose=True)

        if self.loss_func == 'inner':
            print("Defining normalization")
            # Iterate over batches
            loss_norm = 0
            for i in range(0, len(train_dataset), batch_size):
                batch_X, batch_y = train_dataset[i:i+batch_size]
                loss_norm += self.inner_product_loss(batch_X, batch_y[:,model_nos], 0*batch_y[:,model_nos])*len(batch_y)
            self.loss_norm = loss_norm/len(train_dataset)
            
        # Define loss function
        if self.loss_func == 'mse':
            compute_loss = lambda X, outputs, y: self.criterion(outputs, y)
        elif self.loss_func == 'inner':
            compute_loss = lambda X, outputs, y: (self.inner_product_loss(X, outputs, y)/self.loss_norm).sum()
        elif self.loss_func == 'cosine':
            compute_loss = lambda X, outputs, y: self.cosine_loss(X, outputs, y).sum()
        else:
            raise TypeError('Loss function not defined')

        N = len(train_dataset)
        for epoch in range(epochs):

            self.model.train()

            # Define random permutations
            perm = torch.randperm(N)
            
            # Iterate over batches
            epoch_loss = 0
            for i in range(0, N, batch_size):
                idx = perm[i:i+batch_size]
                batch_X, batch_y = train_dataset[idx]
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = compute_loss(batch_X, outputs[:,model_nos], batch_y[:,model_nos])
                
                # Run backprop
                loss.backward()
                self.optimizer.step()
                
                # Update epoch loss
                epoch_loss += loss.item()*len(idx)
            epoch_loss /= N
            train_losses.append(epoch_loss)
            
            if np.isnan(epoch_loss):
                raise Exception("Loss is nan!")

            # Compute validation loss
            val_loss = self.evaluate(val_dataset, model_nos, batch_size=batch_size)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            if epoch%10==0: print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # early stop after some epoch
            if epoch >=0:
                # Save checkpoint if best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(val_loss, epoch, checkpoint_dir)

                # Early stopping check
                if early_stopping(val_loss):
                    print(f"EARLY STOPPING TRIGGERED AT EPOCH {epoch+1}")
                    break
        
        # Load best model
        _, best_loss = self.load_checkpoint(checkpoint_dir)
        print(f"Training completed. Best validation loss: {best_loss:.6f}")

        # Print weight matrix
        print(f"Output weight matrix: {self.model.weights}")
        
        return train_losses, val_losses

    def evaluate(self, dataset, model_nos, batch_size=512):
        self.model.eval()
        total_loss = 0

        N = len(dataset)
        with torch.no_grad():
            for i in range(0, N, batch_size):
                batch_X, batch_y = dataset[i:i+batch_size]
                batch_y = batch_y
                outputs = self.model(batch_X)
                if self.loss_func == 'mse':
                    loss = self.criterion(outputs, batch_y[:,model_nos])
                elif self.loss_func == 'inner':
                    loss = (self.inner_product_loss(batch_X, outputs[:,model_nos], batch_y[:,model_nos])/self.loss_norm).sum()
                elif self.loss_func == 'cosine':
                    loss = self.cosine_loss(batch_X, outputs[:,model_nos], batch_y[:,model_nos]).sum()
                else:
                    raise TypeError('Loss function not defined')
                total_loss += loss.item() * batch_y.size(0)
        return total_loss / N

    def test_loss(self, test_dataset):
        loss = self.evaluate(test_dataset, torch.arange(self.N_models))
        print(f"Test Error: {loss:.6f}")
        return loss

    def get_cosine(self, dataset, dataset2=None, dataset3=None, batch_size=512):
        """
        This computes the cosine between the approximated template and the truth.
        """
        print("Computing cosine...")
        self.model.eval()
        
        # Load true and theory predictions on the combined training+validation+test set
        innerPP, innerTT, innerPT = 0.,0.,0.
        with torch.no_grad():
            # Iterate over all data splits 
            for data in [dataset, dataset2, dataset3]:
                N = len(data)
                for i in range(0, N, batch_size):
                    
                    # Compute theory on all grids
                    batch_X, batch_y = data[i:i+batch_size]
                    outputs = self.model(batch_X)
                    
                    # Compute the inner products
                    k_weight = self.k_weighting(batch_X)[:,None]
                    innerPP += torch.sum(k_weight*outputs**2,axis=0)
                    innerTT += torch.sum(k_weight*batch_y**2,axis=0)
                    innerPT += torch.sum(k_weight*outputs*batch_y,axis=0)
                    
        cosine = (innerPT/torch.sqrt(innerPP*innerTT)).cpu().detach().numpy()
        ratio = (innerPP/innerTT).cpu().detach().numpy()
        return cosine, ratio
        
    def save_model(self, filepath, num_terms, symm_kind, add_bias):
        """
        Save the model and training configuration
        
        Args:
            filepath (str): Path to save the model (e.g., 'models/my_model.pth')
        """
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_terms': num_terms,
            'symm_kind': symm_kind,
            'add_bias': add_bias
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        """
        Load a saved model
        
        Args:
            filepath (str): Path to the saved model
            device (torch.device): Device to load the model on
        
        Returns:
            SepBKNN: Loaded model instance
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create a new instance with the saved parameters
        instance = cls(
            num_terms=checkpoint['num_terms'],
            symm_kind = checkpoint['symm_kind'],
            add_bias = checkpoint['add_bias'],
            device=device,
            )
        
        # Load the saved states
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        instance.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Model loaded from {filepath}")
        return instance
    
    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, min_val=1e-4, verbose=False):
        self.patience = patience
        self.min_delta = min_delta
        self.min_val = min_val
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = float('inf')
    
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.val_loss_min = val_loss
            return False
            
        # Check if validation loss improved or reached threshold
        if (val_loss < self.best_loss - self.min_delta) or (val_loss < self.min_val):
            self.best_loss = val_loss
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                if self.counter%10==0: print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
        
        # Check if we need to stop
        if self.counter >= self.patience:
            self.early_stop = True
            return True
            
        return False