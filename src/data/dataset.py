import torch
from torch.utils.data import Dataset,DataLoader, random_split
import os
class SIPIDataset(Dataset):
    """
     Universal dataset class for all the datasets in the SIPI project. 
     It loads the data from the specified directory and provides an interface to access the data samples.
     Splits Geometric features into Local Targets and Global Targets. Local Targets are the geometric features that are directly related to the design parameters, while Global Targets are the performance metrics that are derived from the geometric features. This is done to ensure the Jacobian Loss is stable with valid inverse generation. The dataset also provides an option to split the data into training and testing sets based on a specified ratio.
    """
    def __init__(self,data_path,dataset_type='link'):
        #load the prepocessed tensors from the specified directory
        data=torch.load(data_path, weights_only=True)
        self.feature_names=data['feature_names']
        
        #define the target names for the local parameters (What the AI generates)
        self.local_features=[
            'VIA_RADIUS', 'PITCH', 'ANTIPAD_RADIUS', 'TMET', 'TDIEL', 
            'CONDUCTIVITY', 'PERMITTIVITY', 'LOSSTANGENT'
        ]
        #SL_WIDTH and LENGTH are added as they are extra columns in the link dataset and are important design parameters.
        if dataset_type=='link':
            self.local_features.extend(['SL_WIDTH']) #trace width is a local SI design choice
        # Define the conditions (Constraints given to the AI - Global Features)
        self.global_features=[
            'VIAS_X_AMOUNT', 'VIAS_Y_AMOUNT', 'SIGNAL_AMOUNT', 
            'GROUND_AMOUNT', 'POWER_AMOUNT'
        ]
        if dataset_type=='link':
            self.global_features.extend(['LENGTH']) #trace length is a global constraint for the AI to meet
        #get column indices for the local and global features for the split
        local_idx = [self.feature_names.index(feature) for feature in self.local_features]
        global_idx = [self.feature_names.index(feature) for feature in self.global_features]
        #split the features into local and global tensors
        self.X_local = data['X'][:, local_idx]
        self.X_global = data['X'][:, global_idx]

        self.Y_real = data['Y_real']
        self.Y_imag = data['Y_imag']

    def __len__(self):
        #returns the total number of samples in the dataset
        return len(self.X_local)
    def __getitem__(self, idx):
        #the dataloader now hands the GPU with targets, global conditions, s-parameters (both real and imag.)
        return self.X_local[idx], self.X_global[idx], self.Y_real[idx], self.Y_imag[idx]
def get_dataloaders(data_path, dataset_type='link', batch_size=32, train_split=0.8, val_split=0.1, seed=42):
    """
    Splits the dataset and retunrs Train, Validation and Test dataloaders. The split is done based on the specified ratios for training, validation and testing. The function also ensures that the split is reproducible by setting a random seed.
    """
    dataset = SIPIDataset(data_path, dataset_type)
    total_samples = len(dataset)
    #calculate the exact number of samples for each split based on the specified ratios
    train_size = int(train_split * total_samples)
    val_size = int(val_split * total_samples)
    test_size = total_samples - train_size - val_size #ensure all samples are allocated to a split and catches any rounding issues
    #lock the random seed for reproducibility of the split
    generator = torch.Generator().manual_seed(seed)

    #randomly split the dataset into training, validation and testing sets based on the calculated sizes
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size], 
        generator=generator
    )

    #create the dataloaders for each split with the specified batch size and shuffling for the training set
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True) #shuffle = True only for training to ensure the model sees the data in a different order each epoch, which can help with generalization
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False) 
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data sucessfully loaded from: {data_path.split('/')[-1]}")
    print(f"Total samples: {total_samples} | Train: {train_size} | Val: {val_size} | Test: {test_size}")
    return train_loader, val_loader, test_loader

#Test to verify batch samples
if __name__ == "__main__":
    #point to the preprocessed dataset (make sure to update the path if needed)
    test_path = os.path.expanduser("~/mece_project_inverse_model/Generative_Inverse_Design_of_High-Speed_Interconnects/data/processed/Universal-Diff-SI-Link/via_link_dataset.pt")
    #grab the loaders
    train_loader, val_loader, test_loader = get_dataloaders(test_path, dataset_type='link', batch_size=32)
    #grab one batch from the training loader to verify the shapes and contents
    X_loc_batch, X_glob_batch, Y_real_batch, Y_imag_batch = next(iter(train_loader))
    print("\nSUCCESS! Scientifically Valid Batch Shapes:")
    print(f"Targets (X_local): {X_loc_batch.shape}  -> (Batch, Local Features)")
    print(f"Conditions (X_global): {X_glob_batch.shape}  -> (Batch, Global Constraints)")
    print(f"S-Params (Y_real): {Y_real_batch.shape} -> (Batch, Freqs, Ports, Ports)")