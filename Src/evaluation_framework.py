# Code written by  @Sungmin

import torch
from torch.nn import DataParallel
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import copy
import torch.nn as nn
from torchsummary import summary
from utils.util_old import *
from utils import loss as loss_lib
import pandas as pd
import os
from utils.eval_utils import total_correlation, compute_modularity, compute_explicitness
from torchvision import models

class Framework(nn.Module):
    def __init__(self, n, z_dim, device, model_type, ga_n, weight_dir, BOE_form = 'BOE'):
        super(Framework, self).__init__()
        self.z = z_dim
        self.model_type = model_type
        print(f'{z_dim=}')
        print(f'{ga_n=}')

        weight_path = os.path.join(weight_dir, "/Saved_models/")

        self.encoder_path=weight_path+"encoder_latest.pth"
        self.decoder_path=weight_path+"decoder_latest.pth"
        self.dis_GAN_path=weight_path+"dis_GAN_latest.pth"

        from models.CCVAEGAN import Encoder, Discriminator, Decoder
        if self.model_type == "VAEGAN":
            self.encoder = Encoder(n, n, z_dim, model='default').to(device)
            self.decoder = Decoder(n, n, int(z_dim/2)).to(device)
            self.dis_GAN = Discriminator().to(device)
        elif self.model_type == "GAVAEGAN":
            self.encoder = Encoder(n, n, z_dim, model='default').to(device)
            self.decoder = Decoder(n, n, int(z_dim/2) + ga_n).to(device)
            self.dis_GAN = Discriminator().to(device)
        elif self.model_type == "CVAEGAN":
            self.encoder = Encoder(n, n, z_dim, model='default').to(device)
            self.decoder = Decoder(n, n, int(z_dim/2) + ga_n).to(device)
            self.dis_GAN = Discriminator().to(device)
        elif self.model_type == "cycleVAEGAN":
            self.encoder = Encoder(n, n, z_dim, model='default').to(device)
            self.decoder = Decoder(n, n, int(z_dim/2) + ga_n).to(device)
            self.dis_GAN = Discriminator().to(device)
        elif self.model_type == "cycleGAVAEGAN":
            self.encoder = Encoder(n, n, z_dim, model='default').to(device)
            self.decoder = Decoder(n, n, int(z_dim/2) + ga_n).to(device)
            self.dis_GAN = Discriminator().to(device)
        elif self.model_type == "CCVAEGAN":
            self.encoder = Encoder(n, n, z_dim, model='default').to(device)
            self.decoder = Decoder(n, n, int(z_dim/2) + ga_n).to(device)
            self.dis_GAN = Discriminator().to(device)
        else:
            raise NameError("Unknown architecture")

    def decode(self, z):
        y = self.decoder(z)
        return y
    
    def encode(self, x):
        z_sample, mu, logvar,_ = self.encoder(x)
        return z_sample, mu, logvar

    def generate(self, x_in, x_ga):
        z, _, _ = self.encode(x_in)
        if self.model_type not in ["VAEGAN", "cycleVAEGAN"]:
            x_gen = self.decode(torch.cat((z, x_ga), 1))
        else:
            x_gen = self.decode(z)
        return x_gen

    def cyclic_recon(self, x_in, x_ga):
        z, _, _ = self.encode(x_in)
        if self.model_type not in ["VAEGAN", "cycleVAEGAN"]:
            x_gen = self.decode(torch.cat((z, x_ga), 1))
        else:
            x_gen = self.decode(z)

        # Cyclic consistency (generated -> original)
        gen_z, _, _ = self.encode(x_gen.detach())
        if self.model_type not in ["VAEGAN", "cycleVAEGAN"]:
            re_gen = self.decode(torch.cat((gen_z.detach(), x_ga), 1))
        else:
            re_gen = self.decode(z)
        return re_gen
    
    def predict_GA(self, x_in):
        _, GA_predict, _= self.dis_GAN(x_in)
        return GA_predict

    def load_weights(self):
        encoder_state_dict = torch.load(self.encoder_path, map_location=self.device)
        decoder_state_dict = torch.load(self.decoder_path, map_location=self.device)
        dis_GAN_state_dict = torch.load(self.dis_GAN_path, map_location=self.device)

        # Extract nested state dict if needed
        if 'encoder' in encoder_state_dict:
            encoder_state_dict = encoder_state_dict['encoder']
        if 'decoder' in decoder_state_dict:
            decoder_state_dict = decoder_state_dict['decoder']
        if 'dis_GAN' in dis_GAN_state_dict:
            dis_GAN_state_dict = dis_GAN_state_dict['dis_GAN']

        self.encoder.load_state_dict(encoder_state_dict)
        self.decoder.load_state_dict(decoder_state_dict)
        self.dis_GAN.load_state_dict(dis_GAN_state_dict)
        print("Weights loaded successfully.")

    def compare_state_dicts(self):
        for model, path in zip([self.encoder, self.decoder, self.dis_GAN], 
                               [self.encoder_path, self.decoder_path, self.dis_GAN_path]):
            original_state_dict = model.state_dict()
            loaded_state_dict = torch.load(path, map_location=self.device)
            
            if 'encoder' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['encoder']
            if 'decoder' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['decoder']
            if 'dis_GAN' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['dis_GAN']

            original_keys = set(original_state_dict.keys())
            loaded_keys = set(loaded_state_dict.keys())

            missing_keys = original_keys - loaded_keys
            unexpected_keys = loaded_keys - original_keys

            if missing_keys:
                print(f'Missing keys in {model.__class__.__name__}: {missing_keys}')
            if unexpected_keys:
                print(f'Unexpected keys in {model.__class__.__name__}: {unexpected_keys}')

            # Check if the parameters are the same
            for key in original_keys & loaded_keys:
                if not torch.equal(original_state_dict[key], loaded_state_dict[key]):
                    print(f'Parameter mismatch at {key} in {model.__class__.__name__}')
        print('State dictionary comparison complete.')

    def check_loaded_keys(self):
        for model, path in zip([self.encoder, self.decoder, self.dis_GAN], 
                               [self.encoder_path, self.decoder_path, self.dis_GAN_path]):
            loaded_state_dict = torch.load(path, map_location=self.device)
            
            if 'encoder' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['encoder']
            if 'decoder' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['decoder']
            if 'dis_GAN' in loaded_state_dict:
                loaded_state_dict = loaded_state_dict['dis_GAN']

            model_state_dict = model.state_dict()

            missing_keys = set(model_state_dict.keys()) - set(loaded_state_dict.keys())
            unexpected_keys = set(loaded_state_dict.keys()) - set(model_state_dict.keys())

            if missing_keys:
                print(f'Missing keys in loaded state dict for {model.__class__.__name__}: {missing_keys}')
            if unexpected_keys:
                print(f'Unexpected keys in loaded state dict for {model.__class__.__name__}: {unexpected_keys}')
            else:
                print(f"All keys matched successfully for {model.__class__.__name__}.")

class Tester:
    def __init__(self, parameters):

        print('')
        print('Testing Model.')
        print('')
        self.device = parameters['device']
        self.ga_n = parameters['ga_n']
        self.model_type = parameters['model_type']  
        self.vmax = parameters['vmax']
        self.output_path = parameters['test_output_path']
        self.save_npy = parameters['save_npy']

        # Construct model and load weights
        self.model = Framework(parameters['slice_size'], parameters['z_dim'], 
                               parameters['device'], parameters['model_type'],
                               parameters['ga_n'], parameters['weight_path']
                               )
        print('Model successfully instanciated...')
        self.model.load_weights()
        self.model.compare_state_dicts()
        self.model.check_loaded_keys()
        print('Model successfully loaded...')

        self.z_dim = parameters['z_dim']
        self.batch = parameters['batch']
        # Establish data loaders
        test_dl = self.test_loader(test_data_path, 'S', self.batch, 158, ga_info)

        self.loader = {"ts": test_dl}
        print('Data loaders successfully loaded...')
        self.image_path = parameters['output_path']
        os.makedirs(output_path, exist_ok=True)

        self.regressor_loss = torch.nn.MSELoss()
        self.l2_loss = torch.nn.MSELoss(reduction='none')
        self.l1_loss = torch.nn.L1Loss(reduction='none')
        self.ssim_map = loss_lib.SSIMComputer()

        # Use a pre-trained ResNet18 and modify it for 1-channel input
        self.feature_extractor = models.resnet18(pretrained=True)
        self.feature_extractor.conv1 = torch.nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.feature_extractor = self.feature_extractor.to(self.device)
        
    def test(self):
            
        # # Setting model for evaluation
        self.model.encoder.eval()
        self.model.decoder.eval()
        self.model.dis_GAN.eval()

        val_MSE = []
        val_MAE = []
        val_SIM = []
        val_MSSIM = []
        
        GT_GA = []
        Pred_GA = []
        Gen_GA = []

        anomaly_score_L1 = []
        anomaly_score_L2 = []
        Subject_ID_total = []

        all_latent_samples = []
        all_feature_samples = []
        all_mu_samples = []
        all_logvar_samples = []
        
        with torch.no_grad():
            for data in self.loader["ts"]:
                val_real = data['image'].to(self.device)
                ga = data['ga'].to(self.device)
                encoded_ga = self.encode_GA(ga, self.ga_n)

                # Run the whole framework forward, no need to do each component separate
                val_gen = self.model.generate(val_real, encoded_ga)
                val_re_gen = self.model.cyclic_recon(val_real, encoded_ga)
                
                val_loss_res_gen_MSE = loss_lib.calc_reconstruction_loss(val_real, val_gen, loss_type="mse", reduction="none")
                val_loss_res_gen_MAE = loss_lib.calc_reconstruction_loss(val_real, val_gen, loss_type="l1", reduction="none")
               
                val_GA_pred_real = self.model.predict_GA(val_real)
                val_GA_pred_gen = self.model.predict_GA(val_gen)
                val_GA_pred_regen = self.model.predict_GA(val_re_gen)
                
                val_MSE.append(val_loss_res_gen_MSE.detach().cpu().numpy())
                val_MAE.append(val_loss_res_gen_MAE.detach().cpu().numpy())

                val_loss_GA_real = self.regressor_loss(val_GA_pred_real, ga)
                val_loss_GA_gen = self.regressor_loss(val_GA_pred_gen, ga)
                val_loss_GA_regen = self.regressor_loss(val_GA_pred_regen, ga)

                GT_GA.append(ga.detach().cpu().numpy())
                Pred_GA.append(val_GA_pred_real.detach().cpu().numpy())
                Gen_GA.append(val_GA_pred_gen.detach().cpu().numpy())

                # Compute error only for brain regions (non-zero area)
                val_gen_batch = self.clipping_th(val_gen.detach().cpu(), 0.015)
                val_real_batch = self.clipping_th(val_real.detach().cpu(), 0.015)
                union_batch = np.array(np.logical_or(val_gen_batch, val_real_batch))
                
                if save_npy:
                    Gen_npy_sample_path = os.path.join(self.output_path, "Gen_sample_npy")
                    os.makedirs(Gen_npy_sample_path, exist_ok=True)
                    np.save("{}/{}.npy".format(Gen_npy_sample_path, data["key"][0]), val_gen_batch)
                
                L1_map_batch = np.array(self.l1_loss(val_gen_batch, val_real_batch))
                L2_map_batch = np.array(self.l2_loss(val_gen_batch, val_real_batch))

                anomaly_score_L1.append(self.masked_mean(L1_map_batch, union_batch))
                anomaly_score_L2.append(self.masked_mean(L2_map_batch, union_batch))
                
                ssim_maps = np.array(self.ssim_map.compute_ssim_map(val_gen_batch, val_real_batch))
                ms_ssim_maps = np.array(self.ssim_map.compute_ms_ssim_map(val_gen_batch, val_real_batch))

                val_SIM.append(self.masked_mean(ssim_maps, union_batch))
                val_MSSIM.append(self.masked_mean(ms_ssim_maps, union_batch))

                Subject_ID_total.append(np.array(data["key"]))

                # feature extraction should be batch-wise
                latent_sample, val_mu, val_logvar =self.model.encode(val_real)
                all_latent_samples.append(latent_sample.cpu().numpy())
                all_feature_samples.append(self.feature_extractor(val_gen).cpu().numpy())
                
                # Total correlation
                all_mu_samples.append(val_mu.cpu().numpy())
                all_logvar_samples.append(val_logvar.cpu().numpy())
                
                # Images dic for visualization
                val_data = {"input": val_real_batch[:,:,:,:], "generation": val_gen_batch[:,:,:,:], 
                            "MAE": L1_map_batch[:,:,:,:], "MSE": L2_map_batch[:,:,:,:], "SSIM": ssim_maps[:,:,:,:], "MSSIM": ms_ssim_maps[:,:,:,:],
                            "key": data['key'], "GA_pred_input": val_GA_pred_real, "GA_pred_gen": val_GA_pred_gen} 
                
                if self.viz_option == "center":
                    self.log(data=val_data, ga = ga.detach(), target_age = ga.detach(), step = 'val', vmax=self.vmax)
                elif self.viz_option == "all":
                    self.log_all(data=val_data, ga = ga.detach(), target_age = ga.detach(), step = 'val', vmax=self.vmax)

            print("Test MAE (mu, std)")            
            print(np.mean(np.array(anomaly_score_L1)), np.std(np.array(anomaly_score_L1)))
            print("Test MSE (mu, std)")            
            print(np.mean(np.array(anomaly_score_L2)), np.std(np.array(anomaly_score_L2)))

            print("Test SIM (mu, std)")            
            print(np.mean(np.array(val_SIM)), np.std(np.array(val_SIM)))
            print("Test MSSIM (mu, std)")            
            print(np.mean(np.array(val_MSSIM)), np.std(np.array(val_MSSIM)))

            # Stack all batches together
            all_latent_samples = np.vstack(all_latent_samples)
            all_feature_samples = np.vstack(all_feature_samples)            
            all_mu_samples = np.vstack(all_mu_samples)            
            all_logvar_samples = np.vstack(all_logvar_samples)            

            # Calculate modularity and explicitness scores
            modularity = compute_modularity(all_latent_samples, all_feature_samples)
            explicitness = compute_explicitness(all_latent_samples, all_feature_samples)            
            TC = total_correlation(all_latent_samples, all_mu_samples, all_logvar_samples)
            
            val_sim_idx = {
                "Val_MSE": val_MSE,
                "Val_MAE": val_MAE,
                "Val_SIM": val_SIM,
                "Val_SSIM": val_MSSIM
            }

            val_DIS_idx = {
                "Modularity": modularity,
                "Explicitness": explicitness,
                "Total_correlation": TC
            }
              
        return val_sim_idx, val_DIS_idx, np.vstack(Subject_ID_total).flatten(), np.vstack(GT_GA), np.vstack(Pred_GA), np.vstack(Gen_GA), np.hstack(anomaly_score_L1), np.hstack(anomaly_score_L2), np.hstack(val_SIM), np.hstack(val_MSSIM)

    def log(self, data, ga, target_age, vmax, step='tr'):
        # Plot and save the progress image
        slide_idx = 0
        for img_idx in range(data["input"].shape[0]):
            if slide_idx==14:
                subject_id = data["key"][img_idx]
                image_to_plot = {key: value[img_idx] for key, value in data.items()}
                progress_im = self.plot4(image_to_plot, ga[img_idx], target_age[img_idx], vmax)
                progress_im.savefig(f'{self.image_path}/{subject_id}_{slide_idx:02}.png')
                plt.close(progress_im)
            slide_idx += 1
            if slide_idx == 30:
                slide_idx = 0

    def log_all(self, data, ga, target_age, vmax, step='tr'):
        # Plot and save the progress image
        slide_idx = 0
        for img_idx in range(data["input"].shape[0]):
            subject_id = data["key"][img_idx]
            image_to_plot = {key: value[img_idx] for key, value in data.items()}
            progress_im = self.plot4(image_to_plot, ga[img_idx], target_age[img_idx], vmax)
            progress_im.savefig(f'{self.image_path}/{subject_id}_{slide_idx:02}.png')
            plt.close(progress_im)
            slide_idx += 1

    def plot(self, data, GA, target_age, vmax):
        
        fig, axs = plt.subplots(1, 4, figsize=(9, 3))
        names = ["input", "generation", "MAE", "MSE"]
        pred_GA_input = data["GA_pred_input"].item()
        pred_GA_gen = data["GA_pred_gen"].item()

        for y in range(4):
            if y == 0:
                axs[y].set_title(f"{names[y]}, GA:{GA.item():.1f}")
                axs[y].imshow(data[names[y]].numpy().squeeze(), cmap='gray')
            elif y == 1:
                axs[y].set_title(f"{names[y]}")
                axs[y].imshow(data[names[y]].numpy().squeeze(), cmap='gray')
            else:
                source = data[names[0]].numpy().squeeze()
                generation = data["generation"].numpy().squeeze()
                union_mask = np.array(np.logical_or(source, generation))

                # Masking: Only show regions where `source` is non-zero or based on a condition
                masked_img = np.where(union_mask != 0, img, 0)  # Set values to 0 where `source` is 0
                im = axs[y].imshow(source, cmap='gray')  # Display the source first

                if y == 2:
                    if vmax=="auto":
                        im = axs[y].imshow(masked_img, cmap='jet', alpha=0.5)  # Overlay the masked image with transparency
                    else:
                        im = axs[y].imshow(masked_img, cmap='jet', vmin=0, vmax=vmax[0], alpha=0.5)  # Overlay the masked image with transparency
                elif y == 3:
                    if vmax=="auto":
                        im = axs[y].imshow(masked_img, cmap='jet', alpha=0.5)  # Overlay the masked image with transparency
                    else:
                        im = axs[y].imshow(masked_img, cmap='jet', vmin=0, vmax=vmax[1], alpha=0.5)  # Overlay the masked image with transparency
                axs[y].set_title(f"{names[y]}")
                # Add colorbar below the specific axes for axs[2] to axs[3]
                if 2 <= y <= 3:
                    cbar = fig.colorbar(im, ax=axs[y], orientation='horizontal', fraction=0.046, pad=0.04)
                    cbar.ax.tick_params(labelsize=8)
            axs[y].axis("off")
        plt.tight_layout()
        return fig 
    
    def encode_GA(self, gas, size):
        # Adjusting the threshold for the nearest 0.1 increment
        threshold_index = size//2
        device = gas.device
        batch_size = gas.size(0)
        # ga_indices = calculate_ga_index_exp(gas, size)
        ga_indices= self.calculate_ga_index(gas, size)
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

    def calculate_ga_index(self, ga, size):
        # Map GA to the nearest increment starting from 20 (assuming a range of 20-40 GA)
        # size * (ga - min_ga) / (max_ga - min_ga)
        increment = (40-20)/size
        ga_mapped = torch.round((ga - 20) / increment)
        
        return ga_mapped  

    def get_center_slice_idx(self, MRI_file):
        nii_file = nib.load(MRI_file, mmap=True)
        # Get the image data as a numpy array
        image_data = nii_file.get_fdata()
        # Access the shape of the image data
        start_idx = round(image_data.shape[2] / 2) - 15
        end_idx = round(image_data.shape[2] / 2) + 15
        ids = np.arange(start=start_idx,stop=end_idx)
        
        return ids
    
    def test_loader(self, source_path, view, batch_size, h, ga_info, raw = False):
        test_id = os.listdir(source_path+'/')
        ids = self.get_center_slice_idx(source_path+'/'+test_id[0])
        test_set = img_dataset(source_path+'/'+test_id[0], view, test_id[0][:-4], size = h, raw = raw, ga_info=ga_info)
        test_set = Subset(test_set,ids)

        for idx,image in enumerate(test_id):
            if idx != 0:
                test_path = source_path + '/' + image
                ts_set = img_dataset(test_path,view, image[:-4], size = h, raw = raw, ga_info=ga_info)
                ids = self.get_center_slice_idx(test_path)
                ts_set = Subset(ts_set,ids)
                test_set = torch.utils.data.ConcatDataset([test_set, ts_set])
                
        # Dataloaders generated from datasets 
        test_final = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=12)
        return test_final
    
    def masked_mean(self, error_maps, mask):
        """
        Computes the mean of the error maps for each batch element, only considering masked regions.

        Args:
            mask (numpy.ndarray): Boolean mask of shape [batch, 1, width, height].
            error_maps (numpy.ndarray): Error maps of shape [batch, 1, width, height].

        Returns:
            numpy.ndarray: Mean value for masked regions of shape [batch].
        """
        batch_size = mask.shape[0]
        means = []
        
        for i in range(batch_size):
            mask_slice = mask[i, 0]  # Shape [width, height] for this batch element
            error_map_slice = error_maps[i, 0]
            masked_error_values = np.where(mask_slice != 0, error_map_slice, 0)  # Set values to 0 where `source` is 0
            sum_value = np.sum(masked_error_values)
            means.append(sum_value/np.sum(mask_slice))

        return np.array(means)  # Shape [batch]
    
