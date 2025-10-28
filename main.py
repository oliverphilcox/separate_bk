import numpy as np, os, argparse, json, datetime, torch
from torch.utils.data import TensorDataset, DataLoader
from sep_bk_nn import SepBKNN
from sep_bk_nn.utils import ConfigHandler, print_block_separator, parse_test_ids
import warnings
import torch.optim as optim

def run_experiment(config, test_id):

    # Use GPUs if available
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using", config['device'])

    # Define inputs
    NUM_TERMS = config['model_params']['num_terms']
    SYMM_KIND = config['model_params']['symm_kind']
    ADD_BIAS = config['model_params']['add_bias']
    print("Model: symmetry type-%d with %d terms"%(SYMM_KIND,NUM_TERMS))
            
    ## Read in external bispectrum
    bk_input = np.loadtxt(config['dataset_params']['datafile'])
    print("Datafile: %s"%config['dataset_params']['datafile'])
    print("N_triangles: %d"%len(bk_input))
    N_models = len(bk_input[0])-3
    print("N_models: %d"%N_models)
    
    # Load k1, k2, k3, (dimensionless) shape
    assert len(bk_input[0])>=4, "External bispectrum must have format {k1,k2,k3,S_1(k1,k2,k3),...}"
    k1, k2, k3 = bk_input[:,:3].T
    S = bk_input[:,3:]
    kmin = np.min([k1,k2,k3])
    kmax = np.max([k1,k2,k3])
    print("kmin = %.2e, kmax = %.2e"%(kmin,kmax))
    
    # Check that all modes obey the triangle inequality
    assert np.all(np.abs(k1-k2)<=k3), "Not all triangles are valid"
    assert np.all(k1+k2>=k3), "Not all triangles are valid"
    
    # Convert to torch
    X_all = torch.tensor(np.stack([k1,k2,k3],axis=1), dtype=torch.float32, device=config['device'])
    #S /= S.std(0)
    y_all = torch.tensor(S, dtype=torch.float32, device=config['device']).view(-1, N_models)
    #print(y_all.mean(0),y_all.std(0))
    
    # Split into train, validation and test set
    all_indices = np.arange(len(X_all))
    np.random.shuffle(all_indices)
    train_indices = all_indices[:len(all_indices)//10*8]
    valid_indices = all_indices[len(all_indices)//10*8:len(all_indices)//10*9]
    test_indices = all_indices[len(all_indices)//10*9:]
    
    X_train = X_all[train_indices]
    y_train = y_all[train_indices]
    X_val = X_all[valid_indices]
    y_val = y_all[valid_indices]
    X_test = X_all[test_indices]
    y_test = y_all[test_indices]
    
    # Print info
    print('Training set size: {}'.format(X_train.shape[0]))
    print('Validation set size: {}'.format(X_val.shape[0]))
    print('Test set size: {}'.format(X_test.shape[0]))

    # Create data loaders
    batch_size = config['batch_size']
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)
    
    # Define path for output model
    model_path = config_handler.get_model_path(test_id)
    results_path = config_handler.get_results_path(test_id)
    intermediate_save_path = config['model_dir'] + '/checkpoints_' + test_id
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    if not os.path.exists(config['model_dir']):
        os.makedirs(config['model_dir'])

    # Define model parameters
    num_terms = config['model_params'].pop('num_terms')
    threshold = config['model_params'].pop('threshold')
    update_weights = config['model_params'].pop('update_weights')
    
    model_nos = torch.arange(N_models)
    old_model = None

    # Iteratively update the number of terms in the representation
    for this_num_terms in range(1,num_terms+1):

        # Define initial class, starting from the last model (if set)
        sep_bk_nn = SepBKNN(**config['model_params'],num_terms=this_num_terms,N_models=N_models,device=config['device'], old_model=old_model)
        
        # Train the model over a number of epochs
        print("\n## Training model with %d terms"%this_num_terms)
        train_losses, val_losses = sep_bk_nn.train(train_dataset, val_dataset, model_nos, batch_size=batch_size, epochs=config['epochs'], checkpoint_dir=intermediate_save_path, patience=config['patience'])

        # Check the outputs
        cosine, ratio = sep_bk_nn.get_cosine(test_dataset, dataset2=train_dataset, dataset3=val_dataset, batch_size=batch_size)
        print('Estimated cosine is {} '.format(cosine))
        print('Estimated ratio is {}'.format(ratio))
        if np.abs(1-cosine)<np.abs(1-threshold):
            print("\n## Threshold reached with %d terms; exiting!"%this_num_terms)
            break

        if num_terms==this_num_terms:
            print("\n## Failed to reach threshold accuracy of %.4f with %d terms"%(threshold, this_num_terms))
            break

        # Store old model
        if update_weights:
            old_model = sep_bk_nn.model
        else:
            print("Resetting weights!")
            old_model = None

    # Test the model by computing the loss on the test set
    test_loss = sep_bk_nn.test_loss(test_dataset)
    
    # Save results
    results = {
        'test_id': test_id,
        'description': config.get('description', ''),
        'timestamp': datetime.datetime.now().isoformat(),
        'test_loss': float(test_loss),
        'cosine': float(cosine.mean()),
        'ratio': float(ratio.mean()),
        'config': config
    }
    
    # Save the model and results
    sep_bk_nn.save_model(model_path, NUM_TERMS, SYMM_KIND, ADD_BIAS)
    with open(f'{results_path}.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Run BK-NN experiments')
    parser.add_argument('--config', type=str, default='configs/experiment_all.yaml',
                      help='Path to config file')
    parser.add_argument('--test_id', type=str, default=None,
                      help='Specific test ID to run (e.g., "test_1" or "1")')
    args = parser.parse_args()
    
    # Load configuration
    global config_handler
    config_handler = ConfigHandler(args.config)
    
    if args.test_id:
        # Run specific experiments
        all_results = {}
        test_ids = parse_test_ids(args.test_id)
        
        print_block_separator(f"Experiments to run: {test_ids}", style='hash')
        
        for test_id in test_ids:
            test_id = str(test_id)
            # Convert number to test_X format if needed
            if test_id.isdigit():
                test_id = f"test_{test_id}"
            
            print_block_separator(f"Starting Experiment: {test_id}")
            print(f"\nRunning experiment {test_id}")
            config = config_handler.get_experiment_config(test_id)
            results = run_experiment(config, test_id)
            all_results[test_id] = results
            
        # Print summary for selected experiments
        print("\nSummary of selected experiments:")
        for test_id, results in all_results.items():
            print(f"{test_id}: LOSS = {results['test_loss']}, COSINE = {results['cosine']}, RATIO = {results['ratio']}")
        print_block_separator("Processing Complete", style='hash')
            
    else:
        # Run all experiments
        all_results = {}
        for test_id in config_handler.get_experiment_ids():
            print_block_separator(f"Starting Experiment: {test_id}")
            print(f"\nRunning experiment {test_id}")
            config = config_handler.get_experiment_config(test_id)
            results = run_experiment(config, test_id)
            all_results[test_id] = results
        
        # Print summary
        print("\nSummary of all experiments:")
        for test_id, results in all_results.items():
            print(f"{test_id}: LOSS = {results['test_loss']}, COSINE = {results['cosine']}, RATIO = {results['ratio']}")

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()