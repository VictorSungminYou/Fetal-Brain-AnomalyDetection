from torch.nn.utils import spectral_norm
import torch.distributions as dist
import torch.nn.functional as F
import torch.nn as nn
import torch

# Author: @Sungmin & @GuillermoTafoya & @simonamador
# The following code builds an autoencoder model for unsupervised learning applications in MRI anomaly detection.
# Preliminary code to condition both GA and Sex
# Sex classifier and Sex embedding are added to CCVAEGAN

def calculate_ga_index(ga, size):
        # Map GA to the nearest increment starting from 20 (assuming a range of 20-40 GA)
        # size * (ga - min_ga) / (max_ga - min_ga)
        increment = (40-20)/size
        ga_mapped = torch.round((ga - 20) / increment)
        return ga_mapped   

def calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, α = 20, β = 40):
        # ( size / exp(a + b) ) * exp(a + ( b * (ga - min_ga) / (max_ga - min_ga) ) )
        ga_mapped = torch.round( (torch.tensor(size) / (torch.exp(torch.tensor(a + b)))) * 
                                (torch.exp(torch.tensor(a) + torch.tensor(b) * ((ga - α) /  (β - α)) )) )
        return ga_mapped  

def inv_calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, α = 20, β = 40):
        ga_mapped = torch.round( (torch.tensor(size) / (torch.exp(torch.tensor(a + b)))) * 
                                (torch.exp(torch.tensor(a) + torch.tensor(b) * ((-ga + β) /  (β - α)) )) )
        return ga_mapped  

def inv_inv_calculate_ga_index_exp(ga, size, a = 1.645, b = 2.688, α = 20, β = 40):
        ψ = calculate_ga_index_exp(40,size, a, b, α, β )
        ga_mapped = - inv_calculate_ga_index_exp(ga,size, a, b, α, β) + ψ
        return ga_mapped 

BOE_forms = {
            'BOE': calculate_ga_index,
            'EBOE': calculate_ga_index_exp,
            'inv_BOE': inv_calculate_ga_index_exp,
            'inv_inv_BOE': inv_inv_calculate_ga_index_exp
        }

def encode_GA(gas, size, BOE_form='BOE'):
        # Adjusting the threshold for the nearest 0.1 increment
        threshold_index = size//2
        device = gas.device
        batch_size = gas.size(0)
        # ga_indices = calculate_ga_index_exp(gas, size)
        ga_indices= BOE_forms[BOE_form](gas, size)
        vectors = torch.full((batch_size, size), -1, device=device)  # Default fill with -1

        for i in range(batch_size):
            idx = ga_indices[i].long()
            if idx > size:
                idx = size
            elif idx < 0:
                idx = 1
            
            if idx >= threshold_index:  # GA >= 30
                new_idx = (idx-threshold_index)*2
                vectors[i, :new_idx] = 1  # First 100 elements to 1 (up to GA == 30)
                vectors[i, new_idx:] = 0  # The rest to 0
            else:  # GA < 30
                new_idx = idx*2
                vectors[i, :new_idx] = 0  # First 100 elements to 0
                # The rest are already set to -1

        return vectors

class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)

class Basic(nn.Module):
    def __init__(self, input, output, k_size=3,stride=1,padding=0,transpose=False):
        super(Basic, self).__init__()

        if transpose == False:
            self.conv_relu_norm = nn.Sequential(
                nn.Conv2d(input, output, k_size, padding=padding,stride=stride),
                nn.LeakyReLU(0.2,inplace=False),
                nn.BatchNorm2d(output)
            )
        else:
            self.conv_relu_norm = nn.Sequential(
                nn.ConvTranspose2d(input, output, k_size, padding=padding,stride=stride),
                nn.LeakyReLU(0.2,inplace=False),
                nn.BatchNorm2d(output)
            )
    def forward(self,x):
        return self.conv_relu_norm(x)


# Encoder class builds encoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the output z-parameters), model (the model type)
class Encoder(nn.Module):
    def __init__(
            self, 
            h,
            w,
            z_dim,
            model: str = 'default',
        ):

        ch = 16
        k_size = 4
        stride = 2
        self.model = model

        super(Encoder,self).__init__()

        self.step0 = Basic(1,ch,k_size=k_size, stride=stride)

        self.step1 = Basic(ch,ch * 2, k_size=k_size, stride=stride)
        self.step2 = Basic(ch * 2,ch * 4, k_size=k_size, stride=stride)
        self.step3 = Basic(ch * 4,ch * 8, k_size=k_size, stride=stride)

        n_h = int(((h-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        n_w = int(((w-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        self.flat_n = n_h * n_w * ch * 8
        self.linear = nn.Linear(self.flat_n, z_dim)

    def forward(self,x):

        embeddings = []

        x = self.step0(x)
        embeddings.append(x)
        x = self.step1(x)
        embeddings.append(x)
        x = self.step2(x)
        embeddings.append(x)
        x = self.step3(x)
        embeddings.append(x)

        x = x.view(-1, self.flat_n)

        z_params = self.linear(x)
        
        mu, log_std = torch.chunk(z_params, 2, dim=1)

        std = torch.exp(log_std)
        z_dist = dist.Normal(mu, std)

        z_sample = z_dist.rsample()

        return z_sample, mu, log_std, {'embeddings': embeddings}

       
# Decoder class builds decoder model depending on the model type.
# Inputs: H, y (x and y size of the MRI slice),z_dim (length of the input z-vector), model (the model type) 
# Note: z_dim in Encoder is not the same as z_dim in Decoder, as the z_vector has half the size of the z_parameters.
class Decoder(nn.Module):
    def __init__(
            self, 
            h, 
            w, 
            z_dim, 
            ):
        super(Decoder, self).__init__()

        self.ch = 16
        self.k_size = 4
        self.stride = 2
        self.hshape = int(((h-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)
        self.wshape = int(((w-self.k_size)/(self.stride**4)) - (self.k_size-1)/(self.stride**3) - (self.k_size-1)/(self.stride**2) - (self.k_size-1)/self.stride + 1)

        self.z_develop = self.hshape * self.wshape * 8 * self.ch
        self.linear = nn.Linear(z_dim, self.z_develop)
        self.step1 = Basic(self.ch* 8, self.ch * 4, k_size=self.k_size, stride=self.stride, transpose=True)
        self.step2 = Basic(self.ch * 4, self.ch * 2, k_size=self.k_size, stride=self.stride, transpose=True)
        self.step3 = Basic(self.ch * 2, self.ch, k_size=self.k_size, stride=self.stride, transpose=True)        
        self.step4 = Basic(self.ch, 1, k_size=self.k_size, stride=self.stride, transpose=True)
        self.activation = nn.Tanh()

    def forward(self,z):
        x = self.linear(z)
        x = x.view(-1, self.ch * 8, self.hshape, self.wshape)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.step4(x)
        recon = self.activation(x)
        return recon

# ----- discriminator -----
class Discriminator(BaseNetwork):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        input_channel = 1
        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(input_channel, 64, 3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=False),
            spectral_norm(nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False))
        )
        
        self.conv_last = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256, 1, 3, stride=1, padding=1)
        )
        
        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(256,1)
        )
        
        self.image_classifer = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256,512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(512,1,1),
            nn.Flatten(),
            nn.Sigmoid()
        )
        
        self.sex_classifier = nn.Sequential(
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(256,512,1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=False),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Conv2d(512,2,1),
            nn.Flatten(),
            nn.Sigmoid()
        )

        #self.init_weights()

    def forward(self, x):
        cov_features = self.conv(x)
        img_features = self.conv_last(cov_features)
        GA_predict = self.regressor(cov_features)
        image_prediction = self.image_classifer(cov_features)
        sex_predict = self.sex_classifier(cov_features)
        return img_features, GA_predict, image_prediction, sex_predict


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        out = self.gamma * out + x
        return out


if __name__ == '__main__':
    from torchsummary import summary
    img_dim = 158
    z_dim = 512 # Image Latent vector 256
    z_sample = int(z_dim/2)
    BOE_GA_dim = 256
    Sex_dim = 2

    emodel = Encoder(img_dim, img_dim, z_dim)
    dmodel = Decoder(h=img_dim, w=img_dim, z_dim=z_sample+BOE_GA_dim+Sex_dim)
    dis_model = Discriminator()


    summary(emodel, (1, img_dim, img_dim), device='cpu')
    summary(dmodel, (1, z_sample+BOE_GA_dim+Sex_dim), device='cpu')
    
    summary(dis_model, (1, img_dim, img_dim), device='cpu')

    input_tensor = torch.randn(1, 1, img_dim, img_dim)
    target_output = torch.randn(1, 1, img_dim, img_dim)

    loss_fn = nn.MSELoss()
    
    output_from_encoder, _, _, _ = emodel(input_tensor) # output_from_encoder is z_sample
    print(f'{output_from_encoder.shape = }')
    assert output_from_encoder.shape == (1, z_sample), "Output shape from encoder is incorrect for decoder input"

    batch_size = output_from_encoder.size(0)

    ga = torch.full((batch_size, 1), 31.0, dtype=torch.float32, device=output_from_encoder.device)
    ga = encode_GA(ga, BOE_GA_dim)
    
    sex = torch.tensor([0,1]).unsqueeze(0)
    
    print(output_from_encoder.shape)
    print(ga.shape)
    print(sex.shape)

    output_from_encoder = torch.cat((output_from_encoder, ga, sex), 1)

    output_from_decoder = dmodel(output_from_encoder)
    print(f'{output_from_decoder.shape  = }')
    print(f'{target_output.shape  = }')
    assert output_from_decoder.shape == target_output.shape, "Output shape from decoder does not match target output shape"

    loss = loss_fn(output_from_decoder, target_output)
    loss.backward()

    for model, name in [(emodel, "Encoder"), (dmodel, "Decoder")]:
        print(f"\nGradients for {name}:")
        for param_name, param in model.named_parameters():
            if param.grad is not None:
                print(f'Gradient of {param_name} after backward pass is valid') # Can also print {param.grad}
            else:
                print(f'No gradient for {param_name}')