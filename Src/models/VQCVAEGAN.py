from torch.nn.utils import spectral_norm
import torch.distributions as dist
import torch.nn.functional as F
import torch.nn as nn
import torch

# Author: @Sungmin & @GuillermoTafoya & @simonamador
# The following code builds an autoencoder model for unsupervised learning applications in MRI anomaly detection.
# This source was under development for potential architecutre update

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

        # n_h = int(((h-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        # n_w = int(((w-k_size)/(stride**4)) - (k_size-1)/(stride**3) - (k_size-1)/(stride**2) - (k_size-1)/stride + 1)
        # self.flat_n = n_h * n_w * ch * 8
        # self.linear = nn.Linear(self.flat_n, z_dim)
        self.last_conv = nn.Conv2d(
                in_channels=ch * 8,
                out_channels=z_dim,
                kernel_size=3,
                padding=1)
        
    def forward(self,x):

        x = self.step0(x)
        x = self.step1(x)
        x = self.step2(x)
        x = self.step3(x)
        x = self.last_conv(x)

        # x = x.view(-1, self.flat_n)

        # z_params = self.linear(x)
        # mu, log_std = torch.chunk(z_params, 2, dim=1)
        # std = torch.exp(log_std)
        # z_dist = dist.Normal(mu, std)
        # z_sample = z_dist.rsample()

        # return z_sample, mu, log_std, {'embeddings': embeddings}
        return x
       
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
            nn.Sigmoid()
        )

        #self.init_weights()

    def forward(self, x):
        cov_features = self.conv(x)
        img_features = self.conv_last(cov_features)
        GA_predict = self.regressor(cov_features)
        image_prediction = self.image_classifer(cov_features)
        return img_features, GA_predict, image_prediction

class SonnetExponentialMovingAverage(nn.Module):
    # See: https://github.com/deepmind/sonnet/blob/5cbfdc356962d9b6198d5b63f0826a80acfdf35b/sonnet/src/moving_averages.py#L25.
    # They do *not* use the exponential moving average updates described in Appendix A.1
    # of "Neural Discrete Representation Learning".
    def __init__(self, decay, shape):
        super().__init__()
        self.decay = decay
        self.counter = 0
        self.register_buffer("hidden", torch.zeros(*shape))
        self.register_buffer("average", torch.zeros(*shape))

    def update(self, value):
        self.counter += 1
        with torch.no_grad():
            self.hidden -= (self.hidden - value) * (1 - self.decay)
            self.average = self.hidden / (1 - self.decay ** self.counter)

    def __call__(self, value):
        self.update(value)
        return self.average
    
class VectorQuantizer(nn.Module):
    def __init__(self, embedding_dim, num_embeddings, use_ema, decay, epsilon):
        super().__init__()
        # See Section 3 of "Neural Discrete Representation Learning" and:
        # https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L142.

        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.use_ema = use_ema
        # Weight for the exponential moving average.
        self.decay = decay
        # Small constant to avoid numerical instability in embedding updates.
        self.epsilon = epsilon

        # Dictionary embeddings.
        limit = 3 ** 0.5
        e_i_ts = torch.FloatTensor(embedding_dim, num_embeddings).uniform_(
            -limit, limit
        )
        if use_ema:
            self.register_buffer("e_i_ts", e_i_ts)
        else:
            self.register_parameter("e_i_ts", nn.Parameter(e_i_ts))

        # Exponential moving average of the cluster counts.
        self.N_i_ts = SonnetExponentialMovingAverage(decay, (num_embeddings,))
        # Exponential moving average of the embeddings.
        self.m_i_ts = SonnetExponentialMovingAverage(decay, e_i_ts.shape)

    def forward(self, x):
        flat_x = x.permute(0, 2, 3, 1).reshape(-1, self.embedding_dim)
        distances = (
            (flat_x ** 2).sum(1, keepdim=True)
            - 2 * flat_x @ self.e_i_ts
            + (self.e_i_ts ** 2).sum(0, keepdim=True)
        )
        encoding_indices = distances.argmin(1)
        quantized_x = F.embedding(
            encoding_indices.view(x.shape[0], *x.shape[2:]), self.e_i_ts.transpose(0, 1)
        ).permute(0, 3, 1, 2)

        # See second term of Equation (3).
        if not self.use_ema:
            dictionary_loss = ((x.detach() - quantized_x) ** 2).mean()
        else:
            dictionary_loss = None

        # See third term of Equation (3).
        commitment_loss = ((x - quantized_x.detach()) ** 2).mean()
        # Straight-through gradient. See Section 3.2.
        quantized_x = x + (quantized_x - x).detach()

        if self.use_ema and self.training:
            with torch.no_grad():
                # See Appendix A.1 of "Neural Discrete Representation Learning".

                # Cluster counts.
                encoding_one_hots = F.one_hot(
                    encoding_indices, self.num_embeddings
                ).type(flat_x.dtype)
                n_i_ts = encoding_one_hots.sum(0)
                # Updated exponential moving average of the cluster counts.
                # See Equation (6).
                self.N_i_ts(n_i_ts)

                # Exponential moving average of the embeddings. See Equation (7).
                embed_sums = flat_x.transpose(0, 1) @ encoding_one_hots
                self.m_i_ts(embed_sums)

                # This is kind of weird.
                # Compare: https://github.com/deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py#L270
                # and Equation (8).
                N_i_ts_sum = self.N_i_ts.average.sum()
                N_i_ts_stable = (
                    (self.N_i_ts.average + self.epsilon)
                    / (N_i_ts_sum + self.num_embeddings * self.epsilon)
                    * N_i_ts_sum
                )
                self.e_i_ts = self.m_i_ts.average / N_i_ts_stable.unsqueeze(0)

        return (
            quantized_x,
            dictionary_loss,
            commitment_loss,
            encoding_indices.view(x.shape[0], -1),
        )


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

class Identifier(nn.Module):
    def __init__(self, num_classes, latent_dim=256):
        super(Identifier, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, num_classes)
        self.softmax = nn.Softmax(dim=1)

        self.residual_connection = nn.Linear(latent_dim, 256)  # To match dimensions
        self._initialize_weights()

    def forward(self, x):
        residual = self.residual_connection(x)  # Apply residual connection
        x = self.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = self.leaky_relu(self.bn2(self.fc2(x)) + residual)
        x = self.dropout2(x)
        x = self.softmax(self.fc3(x))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    from torchsummary import summary
    img_dim = 158
    z_dim = 64 # Latent vector 256
    z_sample = int(z_dim/2)
    BOE_dim = 100

    embedding_dim = 32
    num_embeddings = 32
    decay = 0.5
    use_ema = True
    epsilon = 1e-6
    
    pre_vq_conv = nn.Conv2d(
        in_channels=z_dim, out_channels=embedding_dim, kernel_size=1
    )
    vq = VectorQuantizer(
        embedding_dim, num_embeddings, use_ema, decay, epsilon
    )


    emodel = Encoder(img_dim, img_dim, z_dim)
    dmodel = Decoder2(h=img_dim, w=img_dim, z_dim=z_sample+BOE_dim)

    summary(emodel, (1, img_dim, img_dim), device='cpu')
    summary(dmodel, (1, z_sample+BOE_dim), device='cpu')

    input_tensor = torch.randn(1, 1, img_dim, img_dim)
    target_output = torch.randn(1, 1, img_dim, img_dim)

    loss_fn = nn.MSELoss()
    
    output_from_encoder, _, _, _ = emodel(input_tensor) # output_from_encoder is z_sample
    print(f'{output_from_encoder.shape = }')
    assert output_from_encoder.shape == (1, z_sample), "Output shape from encoder is incorrect for decoder input"

    batch_size = output_from_encoder.size(0)

    ga = torch.full((batch_size, 1), 31.0, dtype=torch.float32, device=output_from_encoder.device)
    ga = encode_GA(ga, BOE_dim)

    output_from_encoder = torch.cat((output_from_encoder,ga), 1)

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

                    
