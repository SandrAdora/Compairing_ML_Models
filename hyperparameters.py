# Create hyperparameters for each modell as a class
import numpy as np

class Hyperparameters:
    
    def __init__(self, model_n):
        self.model_n = model_n
        
    # Get Hyperparameters for chosen model
    def get_params(self):
        """This function creates hyperparameters for the models KNearest Neighbour, Support vector machine and Multilayer Perceptron
        It returns for each model requested it's own initialized hyperparameter for a Search function """
        if self.model_n == 'knn':
            # Titelinformationen für die Konfusionsmatrix
            title = 'K-Nearest Neighbor'
            # Hyperparameter-Grid definieren
            knn_grid = {
                'n_neighbors': np.arange(1, 9, 2), # z.B. [1,3,5,7,9, 11],
                'metric': ['euclidean', 'manhattan', 'minkowski'],     
                'weights': ['uniform', 'distance'],
            }
            return knn_grid, title
        elif self.model_n == 'svm':
            # Titelinformationen für die Konfusionsmatrix
            title = 'Support Vector Machine'
            # Hyperparameter für SVM festlegen
            svm_grid = {
                'kernel': ['linear', 'rbf'], 
                'C': np.logspace(-3, 1, 10), 
            }
            return svm_grid, title
        elif self.model_n == 'mlp':
            # Titelinformationen für die Konfusionsmatrix
            title = 'Neural Network Classifier'
            # Hyperparameter für Neural Network Classifier festlegen
            mlp_grid = {
                'hidden_layer_sizes': [(50, 50), (100, 100), (100, 50), (50, 100)],
                'alpha': np.arange(0.001, 0.01, 0.001),
                'solver': ['adam'], # Für gradientenbasierte Optimierung, lbfgs entfernt, da es nicht konvergierte
                'activation': ['relu', 'logistic', 'sigmoid', 'tangh'], # Reduzierte Anzahl der Aktivierungsfunktionen aus Zeitgründen
                'max_iter': [1000],
                'learning_rate_init': np.arange(0.001, 0.01, 0.001)
            }
            return mlp_grid, title
        else:
            return None, None