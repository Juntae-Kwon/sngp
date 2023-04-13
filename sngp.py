from spectralnorm import spectral_norm
from utils import RandomFourierFeatures
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

# To obtain hidden representations
class snresnet(nn.Module):
    def __init__(self, 
                 num_inputs_features, 
                 num_layers, 
                 num_hidden, 
                 dropout_rate, 
                 num_outputs,
                 spec_norm_bound=0.95, 
                 n_power_iterations=1):
        super().__init__()
        
        self.num_inputs_features = num_inputs_features
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        self.dropout_rate = dropout_rate
        self.num_outputs = num_outputs
        self.spec_norm_bound = spec_norm_bound
        self.n_power_iterations = n_power_iterations

        '''Define structures'''
        # Input Layer
        self.input_layer = nn.Linear(num_inputs_features, num_hidden)
        self.input_layer = spectral_norm(self.input_layer, norm_bound=spec_norm_bound, n_power_iterations=n_power_iterations)
        
        # Hidden Layers
        self.linears = nn.ModuleList(
            [nn.Linear(num_hidden, num_hidden) for i in range(num_layers)]
            )
        for i in range(len(self.linears)):
            self.linears[i] = spectral_norm(self.linears[i], norm_bound=spec_norm_bound, n_power_iterations=n_power_iterations)
        
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):

        # 1st hidden layer by feeding input data to the neural net
        hidden = self.input_layer(inputs)

        # Compute Resnet hidden layers and return output layer
        for resid in self.linears:
            hidden = hidden + self.dropout(F.relu(resid(hidden)))

        return hidden


# To obtain gp output
class Laplace(nn.Module):
    def __init__(self,
                 feature_extractor,
                 num_hidden=128,
                 num_inducing=1024,
                 layer_norm_eps=1e-12,
                 normalize_input=True,
                 scale_random_features=False,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-3,
                 num_classes=1
                 ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.num_hidden = num_hidden
        self.num_inducing = num_inducing
        self.normalize_input = normalize_input
        self.scale_random_features = scale_random_features
        
        self._gp_input_normalize_layer = torch.nn.LayerNorm(num_hidden, eps=layer_norm_eps)
        self._gp_output_layer = nn.Linear(num_inducing, num_classes) # Trainable
        self._random_feature = RandomFourierFeatures(num_hidden, num_inducing) # From the feature extractor
        
        # Laplace Random Feature Covariance
        # Initialize the precision matrix
        self.init_precision_matrix = (self.gp_cov_ridge_penalty * torch.eye(num_inducing)) # s*I
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)

    def gp_layer(self, gp_inputs, update_cov=True):
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs) # normalize inputs
        
        gp_feature = self._random_feature(gp_inputs) 
        gp_feature = torch.cos(gp_feature) # mapped to feature space

        if self.scale_random_features:
            gp_feature = gp_feature * math.sqrt(2. / float(self.num_inducing)) 

        gp_output = self._gp_output_layer(gp_feature)

        if update_cov:
            self.update_cov(gp_feature)  

        return gp_feature, gp_output

    def update_cov(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L346
        """
        GP layer computes the covariance matrix of the random feature coefficient 
        by inverting the precision matrix. 
        """
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = gp_feature.t() @ gp_feature # (num_inducing, num_inducing)
        
        if self.gp_cov_momentum > 0.0:
            # Use moving-average updates to accumulate batch-specific precision matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                self.gp_cov_momentum * self.precision_matrix + (1.0 - self.gp_cov_momentum) * precision_matrix_minibatch
            )
        else:
            # Without momentum
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False) 

    def compute_predictive_covariance(self, gp_feature):
        # https://github.com/google/edward2/blob/main/edward2/tensorflow/layers/random_feature.py#L403
        """
        Compute the predictive covariance matrix:

        s * phi_test @ inv(t(Phi_tr)@Phi_tr + s*I) @ t(Phi_test),
        
        where s is the ridge penalty to be used for stablizing the inverse, and 
        gp_feature represents the random feature of testing data points.

        """
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)
        
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix
    
    def forward(self, x,
                return_gp_cov: bool = False,
                update_cov: bool = True):
        f = self.feature_extractor(x)
        gp_feature, gp_output = self.gp_layer(f, update_cov=update_cov)
        if return_gp_cov:
            gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
            return gp_output, gp_cov_matrix
        return gp_output
    
    def reset_cov(self):
        # For reseting the model's covariance matrix at the beginning of a new epoch.   
        self.precision_matrix = torch.nn.Parameter(copy.deepcopy(self.init_precision_matrix), requires_grad=False)
