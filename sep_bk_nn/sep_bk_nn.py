import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy import interpolate
import os
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
        device = k_input.device

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

        return f_interp

class SepBKNN:
    def __init__(self, symm_kind, num_terms=0, add_bias=False, loss_func='mse', sub_arch='MLP', kpivot=0.05, filterfile=None, log_transform=False, optimal_weights=False, N_models=1, device=None, old_model=None):

        # Store attributes
        self.device = device
        self.N_models = N_models
        self.loss_func = loss_func
        self.sub_arch = sub_arch
        self.num_terms = num_terms
        self.symm_kind = symm_kind
        self.add_bias = add_bias
        self.filterfile = filterfile
        self.kpivot = kpivot
        self.log_transform = log_transform
        self.optimal_weights = optimal_weights

        # Load in separable approximation model, optionally starting from previous version
        if old_model is not None:
            self.model = SeparableApproximation.copy_model(old_model, num_terms).to(self.device)
        else:
            self.model = SeparableApproximation(num_terms=num_terms, N_models=N_models, symm_kind=symm_kind, add_bias=add_bias, sub_arch=sub_arch, log_transform=log_transform, kpivot=kpivot, optimal_weights=optimal_weights).to(self.device)

        # Define loss, optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10, factor=0.3)
        if loss_func == 'mse':
            self.criterion = nn.MSELoss()

        if self.filterfile is not None:
            try:
                self.filter = KFilter(self.filterfile, log_interp=True)
            except IOError:
                raise IOError(f"Error: Filterfile {self.filterfile} not found.")
        else:
            self.filter = lambda k: 1.+0.*k

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
            'add_bias': self.add_bias,
            'optimal_weights': self.optimal_weights
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
        """Compute the squared cosine loss."""
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
            batch_X, batch_y = train_dataset[:]
            loss_norm = self.inner_product_loss(batch_X, batch_y[:,model_nos], 0*batch_y[:,model_nos])*len(batch_y)
            self.loss_norm = loss_norm/len(train_dataset)

        # Define loss function
        if self.loss_func == 'mse':
            compute_loss = lambda X, outputs, y: self.criterion(outputs, y)
        elif self.loss_func == 'inner':
            compute_loss = lambda X, outputs, y: (self.inner_product_loss(X, outputs, y)).sum()/self.loss_norm.sum()
        elif self.loss_func == 'cosine':
            compute_loss = lambda X, outputs, y: self.cosine_loss(X, outputs, y).sum()
        else:
            raise TypeError('Loss function not defined')

        N = len(train_dataset)

        # Preload all training data
        x, y = train_dataset[:N]
        x = x.to(self.device)
        y = y.to(self.device)
        k_weights = self.k_weighting(x)
        k_weights = k_weights.to(self.device)

        for epoch in range(epochs):
            self.model.train()

            # Define random permutations
            if epoch==0: print("batch_size: %d, total_size: %d"%(batch_size,N))
            if batch_size<N:
                perm = torch.randperm(N)

            # Iterate over batches
            epoch_loss = 0
            for i in range(0, N, batch_size):

                if batch_size<N:
                    idx = perm[i:i+batch_size]
                    batch_X, batch_y = train_dataset[idx]
                    batch_weight = k_weights[idx]
                else:
                    batch_X, batch_y = x, y
                    batch_weight = k_weights

                if self.optimal_weights:
                    assert self.loss_func=='inner'

                    # Compute optimal weights
                    basis_terms = self.model.get_predictions(batch_X).T

                    inner_basis_basis = torch.einsum('a,ab,ac->bc',batch_weight,basis_terms,basis_terms)
                    inner_basis_y = torch.einsum('a,ab,ac->bc',batch_weight,basis_terms,batch_y[:,model_nos])
                    inner_y_y = torch.einsum('a,ab,ac->bc',batch_weight,batch_y[:,model_nos],batch_y[:,model_nos])
                    weights = torch.linalg.solve(inner_basis_basis, inner_basis_y)

                    outputs = basis_terms@weights
                else:
                    outputs = self.model(batch_X)

                if self.optimal_weights:
                    inner_sum = torch.diag(inner_y_y) - torch.diag(inner_basis_y.T@weights)
                    loss = (inner_sum/self.loss_norm).sum()/len(outputs[:,model_nos])
                else:
                    loss = compute_loss(batch_X, outputs[:,model_nos], batch_y[:,model_nos])
                loss_fid = loss

                if self.optimal_weights:
                    # Add amplitude penalty
                    basis_sum = torch.sum(torch.diag(inner_basis_basis)**2)
                    if basis_sum>0:
                        penalty = torch.abs(torch.log(basis_sum))
                    else:
                        penalty = 100+0*basis_sum

                    loss_basis = 0.001*penalty
                    loss += loss_basis

                    # Add Fisher matrix penalty
                    if len(inner_basis_basis)>1 and len(model_nos)>1:
                        fish_current = inner_basis_y.T@weights
                        loss_det = torch.sum((fish_current - inner_y_y)**2)/torch.sum(inner_y_y**2)
                        loss += loss_det

                    # Add orthogonalization penalty
                    offdiag = inner_basis_basis - torch.diag(torch.diag(inner_basis_basis))
                    loss_ortho = 0.1*torch.sum(offdiag**2) / torch.sum(torch.diag(inner_basis_basis)**2)+0.01*(torch.logdet(inner_basis_basis)-1)**2
                    loss += loss_ortho

                # Run backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Update epoch loss
                epoch_loss += loss.item()*len(batch_y)
            epoch_loss /= N
            train_losses.append(epoch_loss)

            if np.isnan(epoch_loss):
                raise Exception("Loss is nan!")

            # Compute validation loss
            val_loss = self.evaluate(val_dataset, model_nos, batch_size=batch_size)
            val_losses.append(val_loss)

            self.scheduler.step(val_loss)
            if epoch%10==0: print(f"Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

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

        return train_losses, val_losses

    def evaluate(self, dataset, model_nos, batch_size=512):
        self.model.eval()
        total_loss = 0

        N = len(dataset)
        with torch.no_grad():

            # Load all data
            batch_X, batch_y = dataset[:]

            if self.optimal_weights:
                # Compute optimal weights
                basis_terms = self.model.get_predictions(batch_X).T
                basis_mat = torch.sum(self.k_weighting(batch_X)[:,None,None]*basis_terms[:,:,None]*basis_terms[:,None,:],axis=0)
                basis_dat = torch.sum(self.k_weighting(batch_X)[:,None,None]*basis_terms[:,:,None]*batch_y[:,None,:],axis=0)
                weights = torch.linalg.solve(basis_mat,basis_dat)
                outputs = basis_terms@weights
            else:
                outputs = self.model(batch_X)

            if self.loss_func == 'mse':
                loss = self.criterion(outputs, batch_y[:,model_nos])
            elif self.loss_func == 'inner':
                loss = (self.inner_product_loss(batch_X, outputs[:,model_nos], batch_y[:,model_nos])/self.loss_norm).sum()
            elif self.loss_func == 'cosine':
                loss = self.cosine_loss(batch_X, outputs[:,model_nos], batch_y[:,model_nos]).sum()
            else:
                raise TypeError('Loss function not defined')
            return loss.item()

    def test_loss(self, test_dataset):
        loss = self.evaluate(test_dataset, torch.arange(self.N_models))
        print(f"Test Error: {loss:.6f}")
        return loss

    def get_cosine(self, all_X, all_y):
        """Compute the cosine between the approximated template and the truth."""
        print("Computing cosine...")
        self.model.eval()

        # Load true and theory predictions on the combined training+validation set
        innerPP, innerTT, innerPT = 0.,0.,0.
        with torch.no_grad():

            # Compute theory
            if self.optimal_weights:
                basis_terms = self.model.get_predictions(all_X).T
                inner_basis_basis = torch.sum(self.k_weighting(all_X)[:,None,None]*basis_terms[:,:,None]*basis_terms[:,None,:],axis=0)
                inner_basis_y = torch.sum(self.k_weighting(all_X)[:,None,None]*basis_terms[:,:,None]*all_y[:,None,:],axis=0)
                weights = torch.linalg.solve(inner_basis_basis, inner_basis_y)
                outputs = basis_terms@weights
            else:
                outputs = self.model(all_X)
                weights = None

            # Compute the inner products
            k_weight = self.k_weighting(all_X)[:,None]
            innerPP = torch.sum(k_weight*outputs**2,axis=0)
            innerTT = torch.sum(k_weight*all_y**2,axis=0)
            innerPT = torch.sum(k_weight*outputs*all_y,axis=0)

            # Compute Fisher determinant
            if self.optimal_weights:
                if len(inner_basis_basis)>=len(all_y[0]) and len(all_y[0]>1):
                    fish_true = torch.einsum('a,ab,ac->bc',k_weight[:,0],all_y,all_y)
                    fish_current = inner_basis_y.T@weights
                    print("logdet(F) = %.2f | %.2f"%(torch.logdet(fish_current), torch.logdet(fish_true)))
                    print("Information ratio: %.2f"%(torch.exp(torch.logdet(fish_current)-torch.logdet(fish_true))))

        cosine = (innerPT/torch.sqrt(innerPP*innerTT)).cpu().detach().numpy()
        ratio = (innerPP/innerTT).cpu().detach().numpy()
        return cosine, ratio, weights

    def save_model(self, filepath, num_terms, symm_kind, add_bias, optimal_weights, weights):
        """Save the model and training configuration."""
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'num_terms': num_terms,
            'symm_kind': symm_kind,
            'add_bias': add_bias,
            'optimal_weights': optimal_weights,
        }
        torch.save(save_dict, filepath)
        print(f"Model saved to {filepath}")
        weightpath = filepath[:-4]+'.weight'
        if weights is not None:
            np.save(weightpath, weights.detach().cpu().numpy())
            print(f"Weights saved to {weightpath}")

    @classmethod
    def load_model(cls, filepath, device=None):
        """Load a saved model."""
        checkpoint = torch.load(filepath, map_location=device)
        try:
            checkpoint['optimal_weights']
        except KeyError:
            print("Setting optimal_weights = False")
            checkpoint['optimal_weights'] = False

        # Create a new instance with the saved parameters
        instance = cls(
            num_terms=checkpoint['num_terms'],
            symm_kind = checkpoint['symm_kind'],
            add_bias = checkpoint['add_bias'],
            optimal_weights = checkpoint['optimal_weights'],
            device=device,
            )

        # Load the saved states
        instance.model.load_state_dict(checkpoint['model_state_dict'])
        instance.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        instance.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print(f"Model loaded from {filepath}")
        return instance

    @staticmethod
    def load_weights(filepath):
        """Load a saved set of weights (inferring the weight path from the model path)."""
        return np.load(filepath[:-4]+'.weight.npy')

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
