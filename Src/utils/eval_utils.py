import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torchvision import models

# class DisentanglementEvaluator:
#     def __init__(self, model, latent_dim, feature_extractor=None):
#         """
#         Initialize the evaluator.
        
#         :param model: The generative model (VAE or similar) with encode and decode methods, assumed to be on GPU.
#         :param latent_dim: The dimension of the latent space.
#         :param feature_extractor: A pre-trained feature extractor model (e.g., ResNet). 
#                                   If None, ResNet18 will be used by default.
#         """
#         self.model = model.to('cuda')  # Ensure model is on GPU
#         self.latent_dim = latent_dim
#         if feature_extractor is None:
#             # Use a pre-trained ResNet18 and modify it for 1-channel input
#             self.feature_extractor = models.resnet18(pretrained=True)
#             self.feature_extractor.conv1 = torch.nn.Conv2d(
#                 in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
#             )
#             self.feature_extractor = self.feature_extractor.to('cuda')
#         else:
#             # Assuming custom feature extractor handles 1-channel input correctly
#             self.feature_extractor = feature_extractor.to('cuda')
#         self.feature_extractor.eval()  # Set feature extractor to evaluation mode
    
#     def encode_test_inputs(self, test_data):
#         """
#         Encode the test data into latent space using the model's encoder, ensuring GPU operation.
        
#         :param test_data: A batch of test data samples (e.g., images, text), assumed to be on GPU.
#         :return: Encoded latent vectors for the test data (on GPU).
#         """
#         with torch.no_grad():
#             latent_codes, _, _ = self.model.encode(test_data.to('cuda'))  # Ensure test data is on GPU
#         return latent_codes  # Latent codes are already on GPU
    
#     def generate_reconstructions(self, latent_codes, conditioning_vars):
#         """
#         Generate reconstructed samples from the latent codes and conditioning variables, ensuring GPU operation.
        
#         :param latent_codes: Encoded latent vectors from test data (on GPU).
#         :param conditioning_vars: Conditioning variables for generating samples (on GPU).
#         :return: Generated reconstructions based on the latent vectors (on GPU).
#         """
#         # Ensure both latent codes and conditioning variables are on GPU
#         latent_codes = latent_codes.to('cuda')
#         conditioning_vars = conditioning_vars.to('cuda')
        
#         # Decode the batch of latent codes with the conditioning variables
#         with torch.no_grad():
#             # generated_samples = self.model.decode(latent_codes, conditioning_vars)
#             generated_samples = self.model.decode(torch.cat((latent_codes, conditioning_vars), 1))
        
#         return generated_samples  # Remains on GPU
    
#     def extract_features(self, samples):
#         """
#         Extract high-level features from the generated samples using the feature extractor, ensuring GPU operation.

#         :param samples: A batch of generated samples (1-channel images in channel-first format), assumed to be on GPU.
#         :return: A numpy array of extracted features (moved to CPU).
#         """
#         # Ensure the samples are on the GPU
#         samples = samples.to('cuda')

#         # Check if the input is 1-channel and replicate the channel to 3 channels
#         # if samples.shape[1] == 1:  # Check if it's a batch of 1-channel images
#         #     samples = samples.repeat(1, 3, 1, 1)  # Replicate the single channel to 3 channels

#         with torch.no_grad():
#             # Pass the entire batch through the feature extractor at once
#             features = self.feature_extractor(samples).cpu().numpy()  # Move features to CPU

#         return features
    
#     def modularity_score(self, latent_samples, generated_samples):
#         """
#         Compute the modularity score by measuring the correlation between latent variables 
#         and the high-level features extracted from generated samples, ensuring GPU operation.
        
#         :param latent_samples: The latent vectors corresponding to the test samples (on GPU).
#         :param generated_samples: The reconstructions based on the latent vectors (on GPU).
#         :return: A modularity score (higher is better).
#         """
#         # Extract high-level features from generated samples
#         feature_activations = self.extract_features(generated_samples)
        
#         # Move latent samples back to CPU for correlation calculation
#         latent_samples = latent_samples.cpu().numpy()
        
#         # Compute correlation between latent variables and feature activations
#         correlations = np.corrcoef(latent_samples, feature_activations, rowvar=False)
        
#         # Modularity score based on the independence of latent variables
#         modularity = np.mean(np.abs(correlations))
#         return modularity

#     def modularity_score_rev(self, latent_samples, generated_samples):
#         """
#         Approximate modularity score using correlation between latent variables 
#         and high-level features extracted from generated samples, without ground-truth factors.
        
#         :param latent_samples: Latent vectors corresponding to the test samples (on GPU).
#         :param generated_samples: The reconstructions based on the latent vectors (on GPU).
#         :return: A modularity score (higher is better).
#         """
#         # Extract high-level features from generated samples using a pre-trained model
#         feature_activations = self.extract_features(generated_samples)
        
#         # Move latent samples and feature activations back to CPU for calculation
#         latent_samples = latent_samples.cpu().numpy()
#         feature_activations = feature_activations

#         # Compute correlation between latent variables and feature activations
#         correlations = np.corrcoef(latent_samples.T, feature_activations.T)[:latent_samples.shape[1], latent_samples.shape[1]:]
        
#         # Modularity score based on minimizing correlation overlap between latent variables and features
#         modularity = 0
#         for i in range(correlations.shape[0]):
#             max_corr = np.max(np.abs(correlations[i]))
#             total_corr = np.sum(np.abs(correlations[i]))
#             modularity += max_corr / total_corr  # Penalize overlap with other features

#         modularity /= latent_samples.shape[1]  # Normalize by number of latent variables
#         return modularity  
    
#     def explicitness_score(self, latent_samples, generated_samples):
#         """
#         Compute the explicitness score by training a simple linear regression model 
#         to predict features from latent variables and measuring its accuracy, ensuring GPU operation.
        
#         :param latent_samples: The latent vectors corresponding to the test samples (on GPU).
#         :param generated_samples: The reconstructions based on the latent vectors (on GPU).
#         :return: An explicitness score (higher is better).
#         """
#         # Extract high-level features from generated samples
#         feature_activations = self.extract_features(generated_samples)
        
#         # Move latent samples back to CPU for regression calculation
#         latent_samples = latent_samples.cpu().numpy()
        
#         # Train linear regression to predict features from latent variables
#         regressor = LinearRegression()
#         regressor.fit(latent_samples, feature_activations)
        
#         # Evaluate the model using the R2 score (measure of explained variance)
#         predicted_features = regressor.predict(latent_samples)
#         explicitness_score = r2_score(feature_activations, predicted_features)
        
#         return explicitness_score
    
#     def evaluate(self, test_data, conditioning_vars):
#         """
#         Evaluate both modularity and explicitness using test inputs and conditioning variables, ensuring GPU operation.
        
#         :param test_data: A batch of test data samples (assumed to be on GPU).
#         :param conditioning_vars: Conditioning variables corresponding to test data (assumed to be on GPU).
#         :return: A dictionary containing both modularity and explicitness scores.
#         """
#         # Encode the test inputs into latent space
#         latent_samples = self.encode_test_inputs(test_data)
        
#         # print(test_data.shape)
#         # Generate reconstructions from the latent codes and conditioning variables
#         generated_samples = self.generate_reconstructions(latent_samples, conditioning_vars)
#         # print(generated_samples.shape)

#         # Calculate modularity and explicitness scores
#         modularity = self.modularity_score(latent_samples, generated_samples)
#         explicitness = self.explicitness_score(latent_samples, generated_samples)
        
#         return modularity, explicitness



# class DisentanglementEvaluator_woCond:
#     def __init__(self, model, latent_dim, feature_extractor=None):
#         """
#         Initialize the evaluator.
        
#         :param model: The generative model (VAE or similar) with encode and decode methods, assumed to be on GPU.
#         :param latent_dim: The dimension of the latent space.
#         :param feature_extractor: A pre-trained feature extractor model (e.g., ResNet). 
#                                   If None, ResNet18 will be used by default.
#         """
#         self.model = model.to('cuda')  # Ensure model is on GPU
#         self.latent_dim = latent_dim
#         if feature_extractor is None:
#             # Use a pre-trained ResNet18 and modify it for 1-channel input
#             self.feature_extractor = models.resnet18(pretrained=True)
#             self.feature_extractor.conv1 = torch.nn.Conv2d(
#                 in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
#             )
#             self.feature_extractor = self.feature_extractor.to('cuda')
#         else:
#             # Assuming custom feature extractor handles 1-channel input correctly
#             self.feature_extractor = feature_extractor.to('cuda')
#         self.feature_extractor.eval()  # Set feature extractor to evaluation mode
    
#     def encode_test_inputs(self, test_data):
#         """
#         Encode the test data into latent space using the model's encoder, ensuring GPU operation.
        
#         :param test_data: A batch of test data samples (e.g., images, text), assumed to be on GPU.
#         :return: Encoded latent vectors for the test data (on GPU).
#         """
#         with torch.no_grad():
#             latent_codes, _, _ = self.model.encode(test_data.to('cuda'))  # Ensure test data is on GPU
#         return latent_codes  # Latent codes are already on GPU
    
#     def generate_reconstructions(self, latent_codes):
#         """
#         Generate reconstructed samples from the latent codes and conditioning variables, ensuring GPU operation.
        
#         :param latent_codes: Encoded latent vectors from test data (on GPU).
#         :param conditioning_vars: Conditioning variables for generating samples (on GPU).
#         :return: Generated reconstructions based on the latent vectors (on GPU).
#         """
#         # Ensure both latent codes and conditioning variables are on GPU
#         latent_codes = latent_codes.to('cuda')
#         # Decode the batch of latent codes with the conditioning variables
#         with torch.no_grad():
#             # generated_samples = self.model.decode(latent_codes, conditioning_vars)
#             generated_samples = self.model.decode(latent_codes)
        
#         return generated_samples  # Remains on GPU
    
#     def extract_features(self, samples):
#         """
#         Extract high-level features from the generated samples using the feature extractor, ensuring GPU operation.

#         :param samples: A batch of generated samples (1-channel images in channel-first format), assumed to be on GPU.
#         :return: A numpy array of extracted features (moved to CPU).
#         """
#         # Ensure the samples are on the GPU
#         samples = samples.to('cuda')

#         # Check if the input is 1-channel and replicate the channel to 3 channels
#         # if samples.shape[1] == 1:  # Check if it's a batch of 1-channel images
#         #     samples = samples.repeat(1, 3, 1, 1)  # Replicate the single channel to 3 channels

#         with torch.no_grad():
#             # Pass the entire batch through the feature extractor at once
#             features = self.feature_extractor(samples).cpu().numpy()  # Move features to CPU

#         return features
    
#     def modularity_score(self, latent_samples, generated_samples):
#         """
#         Compute the modularity score by measuring the correlation between latent variables 
#         and the high-level features extracted from generated samples, ensuring GPU operation.
        
#         :param latent_samples: The latent vectors corresponding to the test samples (on GPU).
#         :param generated_samples: The reconstructions based on the latent vectors (on GPU).
#         :return: A modularity score (higher is better).
#         """
#         # Extract high-level features from generated samples
#         feature_activations = self.extract_features(generated_samples)
        
#         # Move latent samples back to CPU for correlation calculation
#         latent_samples = latent_samples.cpu().numpy()
        
#         # Compute correlation between latent variables and feature activations
#         correlations = np.corrcoef(latent_samples, feature_activations, rowvar=False)
        
#         # Modularity score based on the independence of latent variables
#         modularity = np.mean(np.abs(correlations))
#         return modularity
    
#     def modularity_score_rev(self, latent_samples, generated_samples):
#         """
#         Approximate modularity score using correlation between latent variables 
#         and high-level features extracted from generated samples, without ground-truth factors.
        
#         :param latent_samples: Latent vectors corresponding to the test samples (on GPU).
#         :param generated_samples: The reconstructions based on the latent vectors (on GPU).
#         :return: A modularity score (higher is better).
#         """
#         # Extract high-level features from generated samples using a pre-trained model
#         feature_activations = self.extract_features(generated_samples)
        
#         # Move latent samples and feature activations back to CPU for calculation
#         latent_samples = latent_samples.cpu().numpy()
#         feature_activations = feature_activations

#         # Compute correlation between latent variables and feature activations
#         correlations = np.corrcoef(latent_samples.T, feature_activations.T)[:latent_samples.shape[1], latent_samples.shape[1]:]
        
#         # Modularity score based on minimizing correlation overlap between latent variables and features
#         modularity = 0
#         for i in range(correlations.shape[0]):
#             max_corr = np.max(np.abs(correlations[i]))
#             total_corr = np.sum(np.abs(correlations[i]))
#             modularity += max_corr / total_corr  # Penalize overlap with other features

#         modularity /= latent_samples.shape[1]  # Normalize by number of latent variables
#         return modularity    
    
    
#     def explicitness_score(self, latent_samples, generated_samples):
#         """
#         Compute the explicitness score by training a simple linear regression model 
#         to predict features from latent variables and measuring its accuracy, ensuring GPU operation.
        
#         :param latent_samples: The latent vectors corresponding to the test samples (on GPU).
#         :param generated_samples: The reconstructions based on the latent vectors (on GPU).
#         :return: An explicitness score (higher is better).
#         """
#         # Extract high-level features from generated samples
#         feature_activations = self.extract_features(generated_samples)
        
#         # Move latent samples back to CPU for regression calculation
#         latent_samples = latent_samples.cpu().numpy()
        
#         # print(latent_samples.shape)
#         # print(feature_activations.shape)
#         # Train linear regression to predict features from latent variables
#         regressor = LinearRegression()
#         regressor.fit(latent_samples, feature_activations)
        
#         # Evaluate the model using the R2 score (measure of explained variance)
#         predicted_features = regressor.predict(latent_samples)
#         explicitness_score = r2_score(feature_activations, predicted_features)
#         # print(explicitness_score)
        
#         return explicitness_score
    
#     def evaluate(self, test_data, conditioning_vars):
#         """
#         Evaluate both modularity and explicitness using test inputs and conditioning variables, ensuring GPU operation.
        
#         :param test_data: A batch of test data samples (assumed to be on GPU).
#         :param conditioning_vars: Conditioning variables corresponding to test data (assumed to be on GPU).
#         :return: A dictionary containing both modularity and explicitness scores.
#         """
#         # Encode the test inputs into latent space
#         latent_samples = self.encode_test_inputs(test_data)
        
#         # print(test_data.shape)
#         # Generate reconstructions from the latent codes and conditioning variables
#         generated_samples = self.generate_reconstructions(latent_samples)
#         # print(generated_samples.shape)

#         # Calculate modularity and explicitness scores
#         modularity = self.modularity_score(latent_samples, generated_samples)
#         explicitness = self.explicitness_score(latent_samples, generated_samples)
        
#         # print(modularity)
#         # print(explicitness)
#         return modularity, explicitness

import torch
import torch.nn.functional as F

def total_correlation_rep(latent_means, latent_logvars, n_samples=10):
    """
    Estimate total correlation using Monte Carlo sampling with repeated measurements.
    
    :param latent_means: Mean of the latent variables (batch_size, latent_dim)
    :param latent_logvars: Log variance of the latent variables (batch_size, latent_dim)
    :param n_samples: Number of repeated samples to stabilize the estimate.
    :return: Averaged total correlation estimate over multiple samples.
    """
    total_corr_sum = 0.0
    
    for _ in range(n_samples):
        # Sample latent variables from the Gaussian distribution parameterized by mean and log variance
        z_samples = latent_means + torch.exp(0.5 * latent_logvars) * torch.randn_like(latent_logvars)
        
        # Estimate log joint distribution p(z) (log of the joint probability)
        log_qz_given_x = -0.5 * (latent_logvars + torch.pow(z_samples - latent_means, 2) / torch.exp(latent_logvars)).sum(dim=1)

        # Estimate log marginal distribution p(z_i) (log of the marginal probability for each dimension)
        log_qz_i = -0.5 * torch.mean(latent_logvars + torch.pow(z_samples - latent_means, 2) / torch.exp(latent_logvars), dim=0).sum()

        # Total correlation is the difference between log joint and log marginal distributions
        total_corr = torch.mean(log_qz_given_x) - log_qz_i
        
        # Accumulate total correlation across repeated samples
        total_corr_sum += total_corr

    # Return the average total correlation over all repeated samples
    return total_corr_sum / n_samples

# def total_correlation(z_samples, latent_means, latent_logvars):

#     total_corr_sum = 0.0
    
#     # Sample latent variables from the Gaussian distribution parameterized by mean and log variance
#     z_samples = latent_means + torch.exp(0.5 * latent_logvars) * torch.randn_like(latent_logvars)
    
#     # Estimate log joint distribution p(z) (log of the joint probability)
#     log_qz_given_x = -0.5 * (latent_logvars + torch.pow(z_samples - latent_means, 2) / torch.exp(latent_logvars)).sum(dim=1)

#     # Estimate log marginal distribution p(z_i) (log of the marginal probability for each dimension)
#     log_qz_i = -0.5 * torch.mean(latent_logvars + torch.pow(z_samples - latent_means, 2) / torch.exp(latent_logvars), dim=0).sum()

#     # Total correlation is the difference between log joint and log marginal distributions
#     total_corr = torch.mean(log_qz_given_x) - log_qz_i
    
#     # Accumulate total correlation across repeated samples
#     total_corr_sum += total_corr

#     return total_corr_sum

def total_correlation(z_samples, latent_means, latent_logvars):

    total_corr_sum = 0.0
    
    # Sample latent variables from the Gaussian distribution parameterized by mean and log variance
    # z_samples = latent_means + np.exp(0.5 * latent_logvars) * np.random.randn(*latent_logvars.shape)
    
    # Estimate log joint distribution p(z) (log of the joint probability)
    log_qz_given_x = -0.5 * (latent_logvars + np.power(z_samples - latent_means, 2) / np.exp(latent_logvars)).sum(axis=1)

    # Estimate log marginal distribution p(z_i) (log of the marginal probability for each dimension)
    log_qz_i = -0.5 * np.mean(latent_logvars + np.power(z_samples - latent_means, 2) / np.exp(latent_logvars), axis=0).sum()

    # Total correlation is the difference between log joint and log marginal distributions
    total_corr = np.mean(log_qz_given_x) - log_qz_i
    
    # Accumulate total correlation across repeated samples
    total_corr_sum += total_corr

    return total_corr_sum

def extract_features(samples, feature_extractor=None):
    """
    Extract high-level features from the generated samples using a feature extractor.
    
    :param feature_extractor: A pre-trained feature extractor model (e.g., ResNet18).
    :param samples: A batch of generated samples (1-channel images in channel-first format).
    :return: A numpy array of extracted features.
    """
    samples = samples.to('cuda')
    features = feature_extractor(samples).cpu().numpy()  # Extract features and move to CPU
    return features


def compute_modularity(latent_samples, feature_activations):
    """
    Compute the modularity score by measuring the correlation between latent variables 
    and the high-level features extracted from generated samples.
    
    :param latent_samples: Latent vectors corresponding to the test samples.
    :param feature_activations: High-level features extracted from the generated samples.
    :return: A modularity score (higher is better).
    """
    correlations = np.corrcoef(latent_samples, feature_activations, rowvar=False)
    modularity = np.mean(np.abs(correlations))
    return modularity


def compute_explicitness(latent_samples, feature_activations):
    """
    Compute the explicitness score by training a linear regression model 
    to predict features from latent variables and measuring its accuracy.
    
    :param latent_samples: Latent vectors corresponding to the test samples.
    :param feature_activations: High-level features extracted from the generated samples.
    :return: An explicitness score (higher is better).
    """
    regressor = LinearRegression()
    regressor.fit(latent_samples, feature_activations)
    
    predicted_features = regressor.predict(latent_samples)
    explicitness_score = r2_score(feature_activations, predicted_features)
    
    return explicitness_score
