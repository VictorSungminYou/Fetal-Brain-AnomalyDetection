# Code based on https://github.com/ci-ber/PHANES and https://github.com/researchmm/AOT-GAN-for-Inpainting

import monai.losses as losses 
import torch.nn as nn
import torch
import torch.nn.functional as F
from scipy.ndimage.filters import gaussian_filter

from utils.vgg import VGG19

import torch.nn as nn
import torchvision
from torch.nn.modules.loss import _Loss

from skimage.metrics import structural_similarity as ssim
# from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import torch
import numpy as np

def calc_reconstruction_loss(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.sum(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction=reduction)
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error

def calc_reconstruction_loss2(x, recon_x, loss_type='mse', reduction='sum'):
    """
    :param x: original inputs
    :param recon_x:  reconstruction of the VAE's input
    :param loss_type: "mse", "l1", "bce"
    :param reduction: "sum", "mean", "none"
    :return: recon_loss
    """
    if reduction not in ['sum', 'mean', 'none']:
        raise NotImplementedError
    recon_x = recon_x.view(recon_x.size(0), -1)
    x = x.view(x.size(0), -1)
    if loss_type == 'mse':
        recon_error = F.mse_loss(recon_x, x, reduction='none')
        recon_error = recon_error.mean(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'l1':
        recon_error = F.l1_loss(recon_x, x, reduction='none')
        recon_error = recon_error.mean(1)
        if reduction == 'sum':
            recon_error = recon_error.sum()
        elif reduction == 'mean':
            recon_error = recon_error.mean()
    elif loss_type == 'bce':
        recon_error = F.binary_cross_entropy(recon_x, x, reduction=reduction)
    else:
        raise NotImplementedError
    return recon_error

def calc_kl(logvar, mu, mu_o=0.0, logvar_o=0.0, reduce='sum'):
    """
    Calculate kl-divergence
    :param logvar: log-variance from the encoder
    :param mu: mean from the encoder
    :param mu_o: negative mean for outliers (hyper-parameter)
    :param logvar_o: negative log-variance for outliers (hyper-parameter)
    :param reduce: type of reduce: 'sum', 'none'
    :return: kld
    """
    if not isinstance(mu_o, torch.Tensor):
        mu_o = torch.tensor(mu_o).to(mu.device)
    if not isinstance(logvar_o, torch.Tensor):
        logvar_o = torch.tensor(logvar_o).to(mu.device)
    kl = -0.5 * (1 + logvar - logvar_o - logvar.exp() / torch.exp(logvar_o) - (mu - mu_o).pow(2) / torch.exp(
        logvar_o)).sum(1)
    if reduce == 'sum':
        kl = torch.sum(kl)
    elif reduce == 'mean':
        kl = torch.mean(kl)
    return kl

class VGGEncoder(nn.Module):
    """
    VGG Encoder used to extract feature representations for e.g., perceptual losses
    """
    def __init__(self, layers=[1, 6, 11, 20]):
        super(VGGEncoder, self).__init__()
        vgg = torchvision.models.vgg19(pretrained=True).features
        self.encoder = nn.ModuleList()
        temp_seq = nn.Sequential()
        for i in range(max(layers) + 1):
            temp_seq.add_module(str(i), vgg[i])
            if i in layers:
                self.encoder.append(temp_seq)
                temp_seq = nn.Sequential()

    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features




class EmbeddingLoss(torch.nn.Module):
    def __init__(self):
        super(EmbeddingLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
        self.similarity_loss = torch.nn.CosineSimilarity()

    def forward(self, teacher_embeddings, student_embeddings):
        # print(f'LEN {len(output_real)}')
        layer_id = 0
        for teacher_feature, student_feature in zip(teacher_embeddings, student_embeddings):
            if layer_id == 0:
                total_loss = 0.5 * self.criterion(teacher_feature, student_feature)
            else:
                total_loss += 0.5 * self.criterion(teacher_feature, student_feature)
            total_loss += torch.mean(1 - self.similarity_loss(teacher_feature.view(teacher_feature.shape[0], -1),
                                                         student_feature.view(student_feature.shape[0], -1)))
            layer_id += 1
        return total_loss


class PerceptualLoss(_Loss):
    """
    """

    def __init__(
        self,
        reduction: str = 'mean',
        device: str = 'gpu') -> None:
        """
        Args
            reduction: str, {'none', 'mean', 'sum}
                Specifies the reduction to apply to the output. Defaults to ``"mean"``.
                - 'none': no reduction will be applied.
                - 'mean': the sum of the output will be divided by the number of elements in the output.
                - 'sum': the output will be summed.
        """
        super().__init__()
        self.device = device
        self.reduction = reduction
        self.loss_network = VGGEncoder().eval().to(self.device)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        """
        Args:
            input: (N,*),
                where N is the batch size and * is any number of additional dimensions.
            target (N,*),
                same shape as input.
        """
        input_features = self.loss_network(input.repeat(1, 3, 1, 1)) if input.shape[1] == 1 else input
        output_features = self.loss_network(target.repeat(1, 3, 1, 1)) if target.shape[1] == 1 else target

        loss_pl = 0
        for output_feature, input_feature in zip(output_features, input_features):
            loss_pl += F.mse_loss(output_feature, input_feature)
        return loss_pl

def kld_loss(mu, log_var):
    kld = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp())
    return kld

l1_loss = nn.L1Loss()

l1_error = nn.L1Loss(reduction="none")

l2_loss = nn.MSELoss()

ssim_loss = losses.SSIMLoss(win_size=8, spatial_dims=2, reduction="none")

ms_ssim_loss = losses.MultiScaleLoss(loss = losses.SSIMLoss(spatial_dims = 2, win_size = 8, reduction="none"),
                                     scales = [0.5, 1.0, 2.0, 4.0, 8.0], reduction="none")


class SSIMComputer:

    def __init__(self, window_size=7, sigma=1.5, scales=[1.0, 2.0, 4.0, 8.0]):
        """
        Initialize the SSIMComputer with default settings for window size, Gaussian sigma, and scales for MS-SSIM.
        
        Parameters:
        - window_size: The size of the window for SSIM computation.
        - sigma: The sigma value for Gaussian blur applied in MS-SSIM.
        - scales: A list of scale factors for computing MS-SSIM.
        """
        self.window_size = window_size
        self.sigma = sigma
        self.scales = scales

    def compute_ssim_map(self, x, y):
        """
        Compute the full-size SSIM map for each image in the batch.
        
        Parameters:
        - x: First batch of images, shape [batch, 1, height, width] (torch.Tensor).
        - y: Second batch of images, shape [batch, 1, height, width] (torch.Tensor).
        
        Returns:
        - ssim_maps: Full-size SSIM maps, shape [batch, height, width] (torch.Tensor).
        """
        batch_size = x.shape[0]
        ssim_maps = []
        
        for i in range(batch_size):
            img1 = x[i, 0].cpu().numpy()
            img2 = y[i, 0].cpu().numpy()
            _, ssim_map = ssim(
                img1, 
                img2, 
                win_size=self.window_size,
                full=True
            )
            ssim_maps.append(ssim_map[np.newaxis, ...])  # Add a channel dimension
                        
        ssim_maps = np.array(ssim_maps)  # [batch, 1, height, width]
        ssim_maps = torch.tensor(ssim_maps, device=x.device)  # Convert back to torch.Tensor
        return ssim_maps

    def compute_ms_ssim_map(self, x, y):
        """
        Compute the full-size MS-SSIM map for each image in the batch.
        
        Parameters:
        - x: First batch of images, shape [batch, 1, height, width] (torch.Tensor).
        - y: Second batch of images, shape [batch, 1, height, width] (torch.Tensor).
        
        Returns:
        - ms_ssim_maps: Full-size MS-SSIM maps, shape [batch, height, width] (torch.Tensor).
        """
        batch_size, _, height, width = x.shape
        ms_ssim_maps = []

        for i in range(batch_size):
            img1 = x[i, 0].cpu().numpy()
            img2 = y[i, 0].cpu().numpy()
            ssim_maps = []

            for scale_factor in self.scales:
                # Rescale the images based on the scale factor
                scaled_height = int(height * scale_factor)
                scaled_width = int(width * scale_factor)
                img1_resized = resize(img1, (scaled_height, scaled_width), anti_aliasing=True)
                img2_resized = resize(img2, (scaled_height, scaled_width), anti_aliasing=True)

                # Compute SSIM on the scaled images
                _, ssim_map = ssim(img1_resized, img2_resized, win_size=self.window_size, full=True)
                ssim_maps.append(ssim_map)

            # Resize SSIM maps back to the original size and average them
            ms_ssim_map = resize(ssim_maps[0], (height, width), anti_aliasing=True)
            for scale in range(1, len(self.scales)):
                resized_map = resize(ssim_maps[scale], (height, width), anti_aliasing=True)
                ms_ssim_map += resized_map
                # ms_ssim_map *= resized_map
            
            ms_ssim_map /= len(self.scales)
            ms_ssim_maps.append(ms_ssim_map[np.newaxis, ...])  # Add a channel dimension
                    
        ms_ssim_maps = np.array(ms_ssim_maps)  # [batch, 1, height, width]
        ms_ssim_maps = torch.tensor(ms_ssim_maps, device=x.device)  # Convert back to torch.Tensor
        return ms_ssim_maps


class Perceptual(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 1.0, 1.0]):
        super(Perceptual, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()
        self.weights = weights

    def __call__(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        content_loss = 0.0
        prefix = [1, 2, 3, 4, 5]
        for i in range(5):
            content_loss += self.weights[i] * self.criterion(
                x_vgg[f'relu{prefix[i]}_1'], y_vgg[f'relu{prefix[i]}_1'])
        return content_loss

class Style(nn.Module):
    def __init__(self):
        super(Style, self).__init__()
        self.vgg = VGG19().cuda()
        self.criterion = torch.nn.L1Loss()

    def compute_gram(self, x):
        b, c, h, w = x.size()
        f = x.view(b, c, w * h)
        f_T = f.transpose(1, 2)
        G = f.bmm(f_T) / (h * w * c)
        return G

    def __call__(self, x, y):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            y = y.repeat(1, 3, 1, 1)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        style_loss = 0.0
        prefix = [2, 3, 4, 5]
        posfix = [2, 4, 4, 2]
        for pre, pos in list(zip(prefix, posfix)):
            style_loss += self.criterion(
                self.compute_gram(x_vgg[f'relu{pre}_{pos}']), self.compute_gram(y_vgg[f'relu{pre}_{pos}']))
        return style_loss

class smgan():
    def __init__(self): 
        self.loss_fn = nn.MSELoss()
    
    def __call__(self, netD, fake, real, masks): 
        fake_detach = fake.detach()

        g_fake = netD(fake)
        d_fake  = netD(fake_detach)
        d_real = netD(real)

        _, _, h, w = g_fake.size()
        b, c, ht, wt = masks.size()
        
        # Handle inconsistent size between outputs and masks
        if h != ht or w != wt:
            g_fake = F.interpolate(g_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_fake = F.interpolate(d_fake, size=(ht, wt), mode='bilinear', align_corners=True)
            d_real = F.interpolate(d_real, size=(ht, wt), mode='bilinear', align_corners=True)
        d_fake_label = torch.Tensor(gaussian_filter(masks.cpu().detach().numpy(), sigma=1.2)).cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label) * masks / torch.mean(masks)

        return dis_loss.mean(), gen_loss.mean()

class smgan_nomask():
    def __init__(self): 
        self.loss_fn = nn.MSELoss()
    
    def __call__(self, netD, fake, real, ga=None): 
        fake_detach = fake.detach()

        g_fake = netD(fake, ga)
        d_fake  = netD(fake_detach, ga)
        d_real = netD(real, ga)

        # No need to resize or apply masks to align sizes now
        d_fake_label = torch.ones_like(d_fake).cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label)

        return dis_loss.mean(), gen_loss.mean()

# class IdentityLoss:
#     """
#     Calculate a single metric that aims to maximize inter-class KL divergence while minimizing intra-class KL divergence,
#     with class labels provided as one-hot encoded vectors.

#     Args:
#     mu (torch.Tensor): Batch of means for the first set of Gaussian distributions.
#     logvar (torch.Tensor): Batch of log-variances for the first set of Gaussian distributions.
#     mu_gen (torch.Tensor): Batch of means for the second set of Gaussian distributions.
#     logvar_gen (torch.Tensor): Batch of log-variances for the second set of Gaussian distributions.
#     c (torch.Tensor): Batch of one-hot encoded class labels.

#     Returns:
#     float: A single value combining inter-class and intra-class KL divergences.
#     """

#     def __init__(self):
#         pass  # No parameters needed to initialize for this class

#     def __call__(self, mu, logvar, mu_gen, logvar_gen, c):
#         n = len(mu)  # Batch size

#         # Expand dimensions for pairwise comparison
#         mu_expand = mu.unsqueeze(1).expand(-1, n, -1)
#         logvar_expand = logvar.unsqueeze(1).expand(-1, n, -1)
#         mu_gen_expand = mu_gen.unsqueeze(0).expand(n, -1, -1)
#         logvar_gen_expand = logvar_gen.unsqueeze(0).expand(n, -1, -1)

#         # Compute pairwise KL divergence
#         kl_div = 0.5 * (logvar_gen_expand - logvar_expand + 
#                         (torch.exp(logvar_expand) + (mu_expand - mu_gen_expand).pow(2)) / 
#                         torch.exp(logvar_gen_expand) - 1)
#         kl_div = kl_div.sum(2)

#         # Create masks for intra-class and inter-class using one-hot vectors
#         c_expand = c.unsqueeze(1).expand(-1, n, -1)
#         c_gen_expand = c.unsqueeze(0).expand(n, -1, -1)
#         intra_class_mask = (c_expand * c_gen_expand).sum(2).bool()  # Sum across the feature dimension and convert to boolean
#         inter_class_mask = ~intra_class_mask

#         # Calculate intra-class and inter-class KL divergences
#         intra_class_kl_div = kl_div[intra_class_mask].mean()
#         inter_class_kl_div = kl_div[inter_class_mask].mean()

#         # Combine divergences into a single loss metric
#         loss = inter_class_kl_div - intra_class_kl_div

#         return loss

# class IdentityLoss:
#     """
#     Calculate a single metric that aims to maximize inter-class KL divergence while minimizing intra-class KL divergence.

#     Args:
#     mu (torch.Tensor): Batch of means for the first set of Gaussian distributions.
#     logvar (torch.Tensor): Batch of log-variances for the first set of Gaussian distributions.
#     mu_gen (torch.Tensor): Batch of means for the second set of Gaussian distributions.
#     logvar_gen (torch.Tensor): Batch of log-variances for the second set of Gaussian distributions.
#     c (torch.Tensor): Batch of class labels.

#     Returns:
#     float: A single value combining inter-class and intra-class KL divergences.
#     """

#     def __init__(self):
#         pass  # No parameters needed to initialize for this class

#     def __call__(self, mu, logvar, mu_gen, logvar_gen, c):
#         n = len(mu)  # Batch size
#         eps = 1e-8

#         # Expand dimensions for pairwise comparison
#         mu_expand = mu.unsqueeze(1).expand(-1, n, -1)
#         logvar_expand = logvar.unsqueeze(1).expand(-1, n, -1)
#         mu_gen_expand = mu_gen.unsqueeze(0).expand(n, -1, -1)
#         logvar_gen_expand = logvar_gen.unsqueeze(0).expand(n, -1, -1)

#         # Compute pairwise KL divergence
#         kl_div = 0.5 * (logvar_gen_expand - logvar_expand + 
#                         (torch.exp(logvar_expand) + (mu_expand - mu_gen_expand).pow(2)) / 
#                         (torch.exp(logvar_gen_expand) + eps) - 1)
#         kl_div = kl_div.sum(2)

#         # Create masks for intra-class and inter-class
#         c_expand = c.unsqueeze(1).expand(-1, n)
#         c_gen_expand = c.unsqueeze(0).expand(n, -1)

#         intra_class_mask = c_expand == c_gen_expand
#         inter_class_mask = c_expand != c_gen_expand

#         # Calculate intra-class and inter-class KL divergences
#         intra_class_kl_div = kl_div[intra_class_mask].mean()
#         inter_class_kl_div = kl_div[inter_class_mask].mean()

#         # # Combine divergences into a single loss metric
#         loss = inter_class_kl_div - intra_class_kl_div
#         return loss
#         # return inter_class_kl_div, intra_class_kl_div


class IdentityLoss:
    """
    Calculate a single metric that aims to maximize inter-class KL divergence while minimizing intra-class KL divergence.

    Args:
    mu (torch.Tensor): Batch of means for the first set of Gaussian distributions.
    logvar (torch.Tensor): Batch of log-variances for the first set of Gaussian distributions.
    mu_gen (torch.Tensor): Batch of means for the second set of Gaussian distributions.
    logvar_gen (torch.Tensor): Batch of log-variances for the second set of Gaussian distributions.
    c (torch.Tensor): Batch of class labels.

    Returns:
    float: A single value combining inter-class and intra-class KL divergences.
    """

    def __init__(self):
        pass  # No parameters needed to initialize for this class

    def __call__(self, mu, logvar, mu_gen, logvar_gen, c, alpha=0.5):
        n = len(mu)  # Batch size
        eps = 1e-5  # Small constant for numerical stability
        max_clip = 1e5  # Maximum value for clipping

        # Using softplus to ensure positive variance
        var = torch.nn.functional.softplus(logvar) + eps
        var_gen = torch.nn.functional.softplus(logvar_gen) + eps

        # Clip variances to prevent extremely small values
        var = torch.clamp(var, min=eps)
        var_gen = torch.clamp(var_gen, min=eps)

        # Expand dimensions for pairwise comparison
        mu_expand = mu.unsqueeze(1).expand(-1, n, -1)
        mu_gen_expand = mu_gen.unsqueeze(0).expand(n, -1, -1)
        var_expand = var.unsqueeze(1).expand(-1, n, -1)
        var_gen_expand = var_gen.unsqueeze(0).expand(n, -1, -1)

        # Compute pairwise KL divergence safely
        kl_div = 0.5 * ((torch.log(var_gen_expand / var_expand)) +
                        ((var_expand + (mu_expand - mu_gen_expand).pow(2)) / var_gen_expand) - 1)
        kl_div = kl_div.sum(2)

        # Clip KL divergence values to prevent extremely large values
        kl_div = torch.clamp(kl_div, max=max_clip)

        # Handling NaNs and infinities
        kl_div = torch.where(torch.isnan(kl_div) | torch.isinf(kl_div), torch.zeros_like(kl_div), kl_div)

        # Create masks for intra-class and inter-class
        c_expand = c.unsqueeze(1).expand(-1, n)
        c_gen_expand = c.unsqueeze(0).expand(n, -1)

        intra_class_mask = c_expand == c_gen_expand
        inter_class_mask = c_expand != c_gen_expand

        # Calculate intra-class and inter-class KL divergences safely
        if intra_class_mask.any():
            intra_class_kl_div = kl_div[intra_class_mask].mean()
        else:
            intra_class_kl_div = torch.tensor(0.0).to(kl_div.device)

        if inter_class_mask.any():
            inter_class_kl_div = kl_div[inter_class_mask].mean()
        else:
            inter_class_kl_div = torch.tensor(0.0).to(kl_div.device)

        # Combine divergences into a single loss metric
        loss = alpha * inter_class_kl_div - (1 - alpha) * intra_class_kl_div
        return loss
    

class smgan_nomask_feed():
    def __init__(self): 
        self.loss_fn = nn.MSELoss()
    
    def __call__(self, g_fake, d_fake, d_real): 

        # No need to resize or apply masks to align sizes now
        d_fake_label = torch.ones_like(d_fake).cuda()
        d_real_label = torch.zeros_like(d_real).cuda()
        g_fake_label = torch.ones_like(g_fake).cuda()

        dis_loss = self.loss_fn(d_fake, d_fake_label) + self.loss_fn(d_real, d_real_label)
        gen_loss = self.loss_fn(g_fake, g_fake_label)

        return dis_loss.mean(), gen_loss.mean()
    
    
class Identification_loss(nn.Module):
    def __init__(self, num_classes, weight=None, device='cpu'):
        """
        Initialize the cross-entropy loss module with a fixed number of classes.
        :param num_classes: Total number of classes.
        :param weight: A tensor of size `num_classes` containing weights for each class.
                       If None, equal weights are assumed.
        """
        super(Identification_loss, self).__init__()
        self.num_classes = num_classes
        self.device = device

        if weight is None:
            # If no weights are provided, assume uniform weights
            self.weight = torch.ones(num_classes) / num_classes
        else:
            assert weight.numel() == num_classes, "Weight tensor size should match num_classes"
            self.weight = weight
        self.weight = self.weight.to(device)  # Ensure weights are on the right device

    def forward(self, logits, labels):
        """
        Forward pass of the loss calculation.
        :param logits: Logits output from the model (expected size: [batch_size, num_classes]).
        :param labels: Ground truth labels, either as class indices or one-hot encoded labels.
        """
        logits = logits.to(self.device)
        labels = labels.to(self.device)
        # Ensure logits cover all classes
        assert logits.shape[1] == self.num_classes, "Logits must have the same number of classes as initialized"

        # Convert one-hot labels to class indices if necessary
        if labels.dim() == 2 and labels.shape[1] == self.num_classes:
            _, labels = labels.max(dim=1)

        # Calculate the cross-entropy loss using provided class weights
        return F.cross_entropy(logits, labels, weight=self.weight)


