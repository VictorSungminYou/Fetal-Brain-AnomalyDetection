# Code written by  @Sungmin & @GuillermoTafoya & @simonamador

import torch
from torch.nn import DataParallel
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from utils.util import encode_GA, loader
from utils import loss as loss_lib
from time import time
import os

class Framework(nn.Module):
    def __init__(self, n, z_dim, device, model_type, ga_n, BOE_form = 'BOE'):
        super(Framework, self).__init__()
        self.z = z_dim
        self.model_type = model_type
        print(f'{z_dim=}')
        print(f'{ga_n=}')

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
        


class Trainer:
    def __init__(self, parameters):
        print('')
        print('Training Model.')
        print('')

        self.device = parameters['device']
        self.model_path = parameters['model_path']  
        self.tensor_path = parameters['tensor_path'] 
        self.image_path = parameters['image_path']  
        self.ga_n = parameters['ga_n']
        self.model_type = parameters['model_type']  
        

        # Generate model
        self.model = Framework(parameters['slice_size'], parameters['z_dim'], 
                               parameters['device'], parameters['model_type'],
                               parameters['ga_n'])
        print('Model successfully instanciated...')

        self.z_dim = parameters['z_dim']
        self.batch = parameters['batch']
        print('Losses successfully loaded...')

        # Establish data loaders
        train_dl, val_dl = loader(parameters['tr_path'], parameters['val_path'],
                                  parameters['info_path'], parameters['view'], 
                                  parameters['batch'], parameters['slice_size']
                                  )
        self.loader = {"tr": train_dl, "ts": val_dl}
        print('Data loaders successfully loaded...')
        
        # Optimizers
        self.optimizer_e = optim.Adam(self.model.encoder.parameters(), lr=1e-5, weight_decay=1e-6)
        self.optimizer_d = optim.Adam(self.model.decoder.parameters(), lr=1e-5, weight_decay=1e-6)
        self.optimizer_netD = optim.Adam(self.model.dis_GAN.parameters(), lr=1e-4)

        self.scale = parameters['loss_scale']
        self.beta_kl = parameters['beta_kl']
        self.beta_rec = parameters['beta_rec']
        print(f'{parameters["slice_size"]=}')
        print('Optimizers successfully loaded...')

        self.criterion_loss = torch.nn.BCELoss()
        self.regressor_loss = torch.nn.MSELoss()

    def train(self, epochs, b_loss):
        
        # Training Loader
        current_loader = self.loader["tr"]
        
        # Create logger
        self.writer = open(self.tensor_path, 'w')
        self.writer.close()
        self.writer = open(self.tensor_path, 'a')
        self.writer.write('Epoch, tr_ed, tr_g, tr_d, v_ed, v_g, v_d, SSIM, MSE, MAE, Anomaly'+'\n')
        self.best_loss = float("inf") # Initialize best loss (to identify the best-performing model

        epoch_losses = []

        # Trains for all epochs
        for epoch in range(epochs):
            
            # Initialize models in device
            DataParallel(self.model.encoder).to(self.device).train()
            DataParallel(self.model.decoder).to(self.device).train()

            start_time = time()
            # Runs through loader
            for data in current_loader:
 
                images = data['image'].to(self.device)
                ga = data['ga'].to(self.device)
                target_ga = ga
                encoded_ga = encode_GA(ga, self.ga_n)
                encoded_target_ga = encode_GA(target_ga, self.ga_n)
                
                real_batch = images.to(self.device)

                # =========== Update Discriminator ================
                
                for param in self.model.encoder.parameters():
                    param.requires_grad = False
                for param in self.model.decoder.parameters():
                    param.requires_grad = False
                for param in self.model.dis_GAN.parameters():
                    param.requires_grad = True

                # Discriminator loss for real and generated image (0: gen, 1:real)
                real_image_label = torch.ones((self.batch,), dtype=torch.float, device=self.device)
                real_img_features, real_GA_predict, real_image_prediction = self.model.dis_GAN(real_batch)
                real_output = real_image_prediction.view(-1)
                errD_real = self.criterion_loss(real_output, real_image_label)

                real_z, real_mu, real_logvar = self.model.encode(real_batch)
                
                if self.model_type not in ["VAEGAN", "cycleVAEGAN"]:
                    gen = self.model.decoder(torch.cat((real_z.detach(), encoded_target_ga), 1))
                else:
                    gen = self.model.decoder(real_z.detach())                    

                gen_img_features, gen_GA_predict, gen_image_prediction = self.model.dis_GAN(gen.detach())
                gen_output = gen_image_prediction.view(-1)
                gen_image_label = torch.full((self.batch,), 0, dtype=torch.float, device=self.device)
                errD_gen = self.criterion_loss(gen_output, gen_image_label)

                # Compute error of D as sum over the gen and the real batches
                loss_errD = errD_real + errD_gen
                
                # GA prediction loss: Update discriminator based on real image and coressponding GA (CVAEGAN, CCVAEGAN only)
                if self.model_type in ["CVAEGAN", "CCVAEGAN"]:
                    real_loss_GA_regress = self.regressor_loss(real_GA_predict, ga)
                    loss_Dis = loss_errD + real_loss_GA_regress
                else:
                    loss_Dis = loss_errD 
                
                self.optimizer_netD.zero_grad()
                loss_Dis.backward()
                self.optimizer_netD.step()

                # ========= Update Generator ==================
                for param in self.model.encoder.parameters():
                    param.requires_grad = True
                for param in self.model.decoder.parameters():
                    param.requires_grad = True
                for param in self.model.dis_GAN.parameters():
                    param.requires_grad = False

                if self.model_type not in ["VAEGAN", "cycleVAEGAN"]:
                    gen = self.model.decoder(torch.cat((real_z.detach(), encoded_target_ga), 1))
                else:
                    gen = self.model.decoder(real_z.detach())                    

                real_img = real_batch.detach().cpu().numpy().squeeze()
                gen_img = gen.detach().cpu().numpy().squeeze()
                error_img = (gen_img - real_img)

                # Original GAN's discriminator loss (Reversed labeling; 1: gen)
                image_label = torch.ones((self.batch,), dtype=torch.float, device=self.device)
                _, _, gen_image_prediction = self.model.dis_GAN(gen)
                output = gen_image_prediction.view(-1)
                errG_gen = self.criterion_loss(output, image_label)

                if self.model_type in ["cycleVAEGAN", "cycleVAEGAN", "CCVAEGAN"]:
                    # Cyclic consistency (generated -> original)
                    gen_z, gen_mu, gen_logvar = self.model.encode(gen.detach())

                    if self.model_type not in ["VAEGAN", "cycleVAEGAN"]:
                        re_gen = self.model.decoder(torch.cat((gen_z.detach(), encoded_ga), 1))
                    else:
                        re_gen = self.model.decoder(gen_z.detach())                    

                    _, _, re_gen_image_prediction = self.model.dis_GAN(re_gen)
                    re_gen_output = re_gen_image_prediction.view(-1)
                    re_gen_image_label = torch.ones((self.batch,), dtype=torch.float, device=self.device)

                    errG_re_gen = self.criterion_loss(re_gen_output, re_gen_image_label)
                    # Residual loss between real and re_gen
                    loss_res_regen = loss_lib.calc_reconstruction_loss(real_batch, re_gen, loss_type="mse", reduction="mean")
                    # kl_gen = loss_lib.calc_kl(gen_mu, gen_logvar, reduce="mean")
                    # elbo_gen = loss_res_regen + self.beta_kl*kl_gen

                # Residual loss between real and generation
                loss_res_gen = loss_lib.calc_reconstruction_loss(real_batch, gen, loss_type="mse", reduction="mean")
                # KL divergence for real and generated latent variables
                kl_real = loss_lib.calc_kl(real_mu, real_logvar, reduce="mean")
                elbo_real = loss_res_gen + self.beta_kl*kl_real
                
                if self.model_type in ["cycleVAEGAN", "cycleVAEGAN", "CCVAEGAN"]:
                    loss_rec = elbo_real + loss_res_regen
                    loss_errG = errG_gen + errG_re_gen
                else:
                    loss_rec = elbo_real
                    loss_errG = errG_gen

                # GA regression loss
                if self.model_type == "CCVAEGAN":
                    # Cyclic GA regression loss
                    gen_forGA = self.model.decoder(torch.cat((real_z.detach(), encoded_target_ga), 1))
                    # Cyclic consistency (generated -> original)
                    gen_z_forGA, _, _ = self.model.encode(gen_forGA)
                    re_gen_forGA = self.model.decoder(torch.cat((gen_z_forGA, encoded_ga), 1))
                    
                    _, gen_GA_predict, _ = self.model.dis_GAN(gen_forGA)
                    _, re_gen_GA_predict, _ = self.model.dis_GAN(re_gen_forGA)
                    
                    gen_loss_GA_regress = self.regressor_loss(gen_GA_predict, target_ga)
                    re_gen_loss_GA_regress = self.regressor_loss(re_gen_GA_predict, ga)
                    loss_GA_regress = 0.5 * (gen_loss_GA_regress + re_gen_loss_GA_regress)
                elif self.model_type == "CVAEGAN":
                    # GA regression loss
                    gen_forGA = self.model.decoder(real_z.detach())
                    _, gen_GA_predict, _ = self.model.dis_GAN(gen_forGA)
                    gen_loss_GA_regress = self.regressor_loss(gen_GA_predict, target_ga)
                    loss_GA_regress = gen_loss_GA_regress
                
                loss_Gen = self.scale * (loss_errG
                                      + loss_rec * self.beta_rec 
                                      + loss_GA_regress
                                      )

                self.optimizer_e.zero_grad()
                self.optimizer_d.zero_grad()
                loss_Gen.backward()
                self.optimizer_e.step()
                self.optimizer_d.step()
                if torch.isnan(loss_Gen):
                    print('is nan for D')
                    raise SystemError

            # ====================================
            images = {"input": real_img, "prediction": gen_img, "error": error_img} 
            
            print("="*50)
            print('Epoch: {} \t Generator Loss: {:.6f}, Discriminator Loss: {:.6f}'.format(
                epoch, loss_Gen, loss_Dis))
            print("-"*20+" Training "+"-"*20)
            print("[Discriminator losses]")
            if self.model_type in ["CVAEGAN", "CCVAEGAN"]:
                print({"Train/Loss_real_GA_regress": real_loss_GA_regress.item()})

            print({"Train/Loss_errD": loss_errD.item()})
            print({"    Train/errD_real": errD_real.item()})            
            print({"    Train/errD_fake": errD_real.item()})            
            print("-"*50)
            print("[Generator losses]")
            print({"Train/Loss_errG": loss_errG.item()})
            print({"    Train/Loss_errG_gen": errG_gen.item()})
            if self.model_type in ["cycleVAEGAN", "cycleGAVAEGAN", "CCVAEGAN"]:
                print({"    Train/Loss_errG_re_gen": errG_re_gen.item()})

            print({"Train/Loss_rec": loss_rec.item()})
            print({"    Train/loss_res_gen": loss_res_gen.item()})
            if self.model_type in ["cycleVAEGAN", "cycleGAVAEGAN", "CCVAEGAN"]:
                print({"    Train/loss_res_regen": loss_res_regen.item()})

            if self.model_type in ["CVAEGAN", "CCVAEGAN"]:
                print({"Train/Loss_GA_regress": loss_GA_regress.item()})
                print({"    Train/Loss_gen_GA_regress": gen_loss_GA_regress.item()})
                if self.model_type == "CCVAEGAN":
                    print({"    Train/Loss_re_gen_GA_regress": re_gen_loss_GA_regress.item()})
            print("="*50)

            losses = {
                "Loss_Gen": loss_Gen,
                "Loss_real_GA_regress": real_loss_GA_regress,
                "Loss_gen_GA_regress": loss_GA_regress,
                "Loss_Discriminator": loss_Dis,
                "Loss_errG": loss_errG,
            }

            # # Assuming you have variables `current_epoch`, `total_epochs`, `current_val_loss`, and `images` defined:
            self.log(epoch=epoch, epochs=epochs, losses=losses, images=images, ga = ga.detach(), target_age = target_ga.detach(), n_plots = 5, step = 'tr')

            # Validation
            # # Setting model for validation model
            self.model.encoder.eval()
            self.model.decoder.eval()
            self.model.dis_GAN.eval()

            b_val_loss_res_gen, b_val_loss_res_regen = 0.0, 0.0
            b_val_loss_GA_real, b_val_loss_GA_gen, b_val_loss_GA_regen = 0.0, 0.0, 0.0
            
            with torch.no_grad():
                for data in self.loader["ts"]:
                    val_real_batch = data['image'].to(self.device)
                    ga = data['ga'].to(self.device)
                    encoded_ga = encode_GA(ga, self.ga_n)

                    # Run the whole framework forward, no need to do each component separate
                    val_gen = self.model.generate(val_real_batch, encoded_ga)
                    val_re_gen = self.model.cyclic_recon(val_real_batch, encoded_ga)

                    val_gen_img = val_gen.detach().cpu().numpy().squeeze()
                    val_real_img = val_real_batch.detach().cpu().numpy().squeeze()
                    val_error_img = (val_gen_img - val_real_img)

                    val_loss_res_gen = loss_lib.calc_reconstruction_loss(val_real_batch, val_gen, loss_type="mse", reduction="mean")
                    val_loss_res_regen = loss_lib.calc_reconstruction_loss(val_real_batch, val_re_gen, loss_type="mse", reduction="mean")
                    
                    val_GA_pred_real =  self.model.predict_GA(val_real_batch)
                    val_GA_pred_gen = self.model.predict_GA(val_gen)
                    val_GA_pred_regen = self.model.predict_GA(val_re_gen)

                    val_loss_GA_real = self.regressor_loss(val_GA_pred_real, ga)
                    val_loss_GA_gen = self.regressor_loss(val_GA_pred_gen, ga)
                    val_loss_GA_regen = self.regressor_loss(val_GA_pred_regen, ga)

                    b_val_loss_res_gen += val_loss_res_gen
                    b_val_loss_res_regen += val_loss_res_regen
                    b_val_loss_GA_real += val_loss_GA_real
                    b_val_loss_GA_gen += val_loss_GA_gen
                    b_val_loss_GA_regen += val_loss_GA_regen

                b_val_loss_res_gen /= len(self.loader["ts"])
                b_val_loss_res_regen /= len(self.loader["ts"])
                b_val_loss_GA_real /= len(self.loader["ts"])
                b_val_loss_GA_gen /= len(self.loader["ts"])
                b_val_loss_GA_regen /= len(self.loader["ts"])

                # Images dic for visualization
                val_images = {"input": val_gen_img, "prediction": val_real_img, "error": val_error_img} 

                val_loss = {
                    "Val_Loss_Gen": b_val_loss_res_gen,
                    "Val_Loss_ReGen": b_val_loss_res_regen,
                    "val_loss_GA_real": b_val_loss_GA_real,
                    "Val_loss_GA_gen": b_val_loss_GA_gen,
                    "Val_loss_GA_regen": b_val_loss_GA_regen
                }

            print("-"*20+" Validation "+"-"*20)
            print({"Val_Loss_Gen": b_val_loss_res_gen.item()})
            print({"Val_loss_GA_real": b_val_loss_GA_real.item()})
            print({"Val_loss_GA_gen": b_val_loss_GA_gen.item()})
            print({"Val_loss_GA_regen": b_val_loss_GA_regen.item()})            
            print("="*50)
            
            self.log(epoch=epoch, epochs=epochs, losses=val_loss, images=val_images, ga = ga.detach(), target_age = target_ga.detach(), n_plots = 5, step = 'val')

        self.writer.close()

    def log(self, epoch, epochs, losses, images, ga, target_age, n_plots = 1, step='tr'):
        components = ['encoder', 'decoder', 'dis_GAN']
        for component in components:
            torch.save({
                'epoch': epoch + 1,
                component: getattr(self.model, component).state_dict(),
            }, f'{self.model_path}/{component}_latest.pth')

        if epoch == 9999:
            n_plots = 50
            
        n_plots = min(n_plots, images['input'].shape[0])  # Assuming 'input' key exists
        random_indices = np.random.choice(images['input'].shape[0], n_plots, replace=False)
        
        # Save and plot model components every n epochs or in the first or last epoch
        n = 1000
        if (epoch == 0) or ((epoch + 1) % n == 0) or ((epoch + 1) == epochs):
            for component in components:
                torch.save({'epoch': epoch + 1, component: getattr(self.model, component).state_dict()},
                        f'{self.model_path}/{component}_{epoch + 1}.pth')
            # Plot and save the progress image
            # for img_idx in range(image.shape[0]):
            for img_idx in random_indices:
                image_to_plot = {key: value[img_idx] for key, value in images.items()}
                progress_im = self.plot(image_to_plot, ga[img_idx], target_age[img_idx])
                progress_im.savefig(f'{self.image_path}{step}_epoch_{epoch+1}_{img_idx}.png')

        if step == 'val':
            # Save the best model components if current validation loss is lower than the best known loss (Not used in the paper)
            primary_val_loss = losses["Val_Loss_Gen"]    # Default is reconstruction loss (input vs generation)
            
            if isinstance(primary_val_loss, torch.Tensor):
                primary_val_loss_value = primary_val_loss.item()  # Convert tensor to a Python number
            else:
                primary_val_loss_value = primary_val_loss  # It's already a number, not a tensor
            if primary_val_loss_value < self.best_loss:
                self.best_loss = primary_val_loss
                for component in components:
                    torch.save({
                        'epoch': epoch + 1,
                        component: getattr(self.model, component).state_dict(),
                    }, f'{self.model_path}/{component}_best.pth')
                print(f'Saved best model components at epoch {epoch+1} with primary validation loss {primary_val_loss_value:.4f}')

    def plot(self, images, ga, target_age):
        fig, axs = plt.subplots(1, 3, figsize=(10, 6))
        names = ["input", "prediction", "error"]
        for y in range(3):
            axs[y].imshow(images[names[y]], cmap='gray')
            if y == 1:
                axs[y].set_title(f"{names[y]}, GA: {target_age.item():.1f}")
            elif y == 2:
                axs[y].set_title(f"{names[y]}")
            axs[y].axis("off")
        plt.tight_layout()
        
        return fig