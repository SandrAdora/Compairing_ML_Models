import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import pandas as pd

dataset = pd.read_csv('diabetes.csv')

class CreateDataset:
    
    def __init__(self, data=dataset):
        self.dataset = data
        self.scalers = {
            'norm': MinMaxScaler(), # For MinMax normalization 
            'standard': StandardScaler(), # For Standardization
}
        
    # HELP-Function: Handle the Label Class 
    def last_column(self, data=dataset):
        """This function returns the label class from the dataset"""
        y = data['Outcome']
        return y
        
    # Create a dataset     
    def create_dataset(self):
        """This function creates a dataset from the default dataset and returns a featurematrx and a label array"""
            # Check if dataset is complete, if not complete complete dataset by setting the last column
        if len(self.dataset.columns) < 9:
            X = self.dataset.iloc[:,:]
            y = self.last_column() # The original dataset Outcome column
            return X, y
        else:
            X = self.dataset.iloc[:,:-1]
            y = self.dataset['Outcome']
            return X,y
            
    # Create copies from the original Dataset
    def dataset_copy(self):
        """This function creates a copy of the dataset"""
        copy = self.dataset.copy()
        return copy
        
    # Create Datasettypes
    def Dataset_types(self, name):
        """This function changes the datatypes to prefered type
        Parameters:
            name: Normalization || Standardization or MinMax
            self.dataset: Dataset
         """
        features_ = []
        if len(self.dataset.columns) > 8:
            features_ = self.dataset.iloc[:, :-1]
        else:
            features_ = self.dataset.iloc[:, :]
        # Change values in the column to float then cp them to different datasets
        df_floated = features_.values.astype(float)
        dataset_1 = df_floated.copy()
        dataset_2 = df_floated.copy()
        
        if name == "norm":
            print('You chose...MinMaxing...')
            # Use MinMax Scaler when given name is norm and return the changed dataset
            minMax = MinMaxScaler()
            df_minMax_scaled = minMax.fit_transform(dataset_1)
            minMax = pd.DataFrame(df_minMax_scaled, columns=features_.columns)
            return minMax   
        elif name == "standard":
            print("You chose...Standardizing...")
            # Using Standard Scaler when given name is standard and return the changed dataset
            standardScaler = StandardScaler()
            df_standard = standardScaler.fit_transform(dataset_2)
            standardized = pd.DataFrame(df_standard, columns=features_.columns)
            return standardized  
        else:
            raise ValueError('Scaler could not be found')