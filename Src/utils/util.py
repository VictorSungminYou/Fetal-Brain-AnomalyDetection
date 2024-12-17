import argparse
import os, time
from collections import OrderedDict
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import imutils
import operator
import os
import csv
import nibabel as nib
from functools import lru_cache

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


class DatasetSingleton:
    _instance = None
    ga_dict = {}
    sex_dict = {}

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(DatasetSingleton, cls).__new__(cls)
            cls._initialize_data(*args, **kwargs)
        return cls._instance

    @classmethod
    def _initialize_data(cls, csv_paths):
        for data_type, path in csv_paths.items():
            with open(path, 'r') as csvfile:
                csvreader = csv.DictReader(csvfile)
                for row in csvreader:
                    key = row['ID']
                    if data_type == 'ga':
                        cls.ga_dict[key] = float(row['GA'])
                    elif data_type == 'sex':
                        if row['Sex']=='Male':
                            cls.sex_dict[key] = [1,0]
                        elif row['Sex']=='Female':
                            cls.sex_dict[key] = [0,1]
                        elif row['Sex']=='Unknown':
                            cls.sex_dict[key] = [0,0]

class img_dataset(Dataset):

    # Begin the initialization of the datasets. Creates dataset iterativey for each subject and
    # concatenates them together for both training and testing datasets (implements img_dataset class).
    def __init__(self, root_dir, view, key, info_file, data = 'healthy', size: int = 158, horizontal_flip: bool = False, 
                 vertical_flip: bool = False, rotation_angle: int = None):
        self.root_dir = root_dir
        self.view = view
        self.horizontal = horizontal_flip
        self.vertical = vertical_flip
        self.angle = rotation_angle
        self.size = size
        self.key = key
        self.info_file = info_file
        if 'recon' in self.key:
            self.key = self.key[:self.key.index('recon')-1]
        self.data = data
        self.dataset_singleton = DatasetSingleton({
            'ga': info_file,
            'sex': info_file
        })

    def __len__(self):
        view_sizes = {'C': 158, 'A': 158, "S": 158}
        return view_sizes.get(self.view, 0)

    def extract_sex(self):
        sex = self.dataset_singleton.sex_dict.get(self.key)
        if sex is None:
            print(f"Sex not found for {self.key}")
            raise Exception(f"Sex not found for {self.key}")
        
        sex = torch.tensor(sex).type(torch.float)
        return sex
        
    def extract_age(self):
        ga = self.dataset_singleton.ga_dict.get(self.key)
        if ga is None:
            print(f"GA not found for {self.key}")
            raise Exception(f"GA not found for {self.key}")
        
        ga = np.expand_dims(ga, axis=0)
        ga = torch.tensor(ga).type(torch.float)
        return ga
    
    def rotation(self, x, alpha):
        y = x.astype(np.uint8)
        y_rot = imutils.rotate(y, angle = alpha)
        return y_rot.astype(np.float64)
    
    def resizing(self, img, n):
        target = (n, n)
        if (img.shape > np.array(target)).any():
            target_shape2 = np.min([target, img.shape],axis=0)
            start = tuple(map(lambda a, da: a//2-da//2, img.shape, target_shape2))
            end = tuple(map(operator.add, start, target_shape2))
            slices = tuple(map(slice, start, end))
            img = img[tuple(slices)]
        offset = tuple(map(lambda a, da: a//2-da//2, target, img.shape))
        slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
        result = np.zeros(target)
        result[tuple(slices)] = img
        return result
    
    def adjust_image(self, img, target_size):
        from scipy.ndimage import center_of_mass, shift

        original_size = np.array(img.shape)
        padding_needed = np.array(target_size) - original_size
        padding_needed = np.maximum(padding_needed, 0)
        padding_before = padding_needed // 2
        padding_after = padding_needed - padding_before

        # Apply padding
        padded_img = np.pad(img, ((padding_before[0], padding_after[0]), (padding_before[1], padding_after[1])), 'constant', constant_values=0)

        brain_mask = padded_img > 0
        brain_com = center_of_mass(brain_mask)
        padded_center = np.array(padded_img.shape) / 2
        shift_amount = padded_center - np.array(brain_com)

        shifted_img = shift(padded_img, shift=shift_amount)

        current_size = np.array(shifted_img.shape)
        crop_needed = current_size - np.array(target_size)
        crop_before = crop_needed // 2
        crop_after = crop_needed - crop_before
        # Ensure cropping indices are non-negative
        crop_before = np.maximum(crop_before, 0)
        crop_after = np.maximum(crop_after, 0)
        # Apply cropping
        final_img = shifted_img[crop_before[0]:current_size[0]-crop_after[0], crop_before[1]:current_size[1]-crop_after[1]]

        return final_img

    def normalize_95(self, x):
        p98 = np.percentile(x, 99)
        num = x-np.min(x)
        den = p98-np.min(x)
        out = np.zeros((x.shape[0], x.shape[1]))

        x = np.divide(num, den, out=out, where=den!=0)
        return x.clip(0, 1)
    

    def clip_and_resize(self, img, n, percentile=50):
        
        # Continue with the existing resizing process
        target = (n, n)
        if (img.shape > np.array(target)).any():
            target_shape2 = np.min([target, img.shape], axis=0)
            start = tuple(map(lambda a, da: a//2 - da//2, img.shape, target_shape2))
            end = tuple(map(operator.add, start, target_shape2))
            slices = tuple(map(slice, start, end))
            img = img[tuple(slices)]
        offset = tuple(map(lambda a, da: a//2 - da//2, target, img.shape))
        slices = [slice(offset[dim], offset[dim] + img.shape[dim]) for dim in range(img.ndim)]
        result = np.zeros(target)
        result[tuple(slices)] = img
        # Calculate the threshold intensity for the specified percentile
        threshold = np.percentile(result, percentile)
        
        # Set all pixels below this threshold to zero
        result[result < threshold] = 0
        return result

    @lru_cache(maxsize=10000)
    def __getitem__(self, idx):
        nii_file = nib.load(self.root_dir, mmap=True)
        # Get the image data as a numpy array
        image_data = nii_file.get_fdata()
        # Access the shape of the image data
        start_idx = round(image_data.shape[2] / 2) - 15
        end_idx = round(image_data.shape[2] / 2) + 15
        
        slice_data = np.array(nii_file.dataobj[:, :, start_idx:end_idx])
        
        if self.view == 'S':
            slice_data = nii_file.dataobj[idx, :, :]
        elif self.view == 'C':
            slice_data = nii_file.dataobj[:, idx, :]
        else:
            slice_data = nii_file.dataobj[:, :, idx]
        
        slice_data = np.array(slice_data)
        
        n_img = self.clip_and_resize(slice_data, self.size, percentile=10)
        n_img = self.normalize_95(n_img)

        if self.horizontal:
            n_img = np.flip(n_img, axis=0)
        if self.vertical:
            n_img = np.flip(n_img, axis=1)
        if self.angle is not None:
            n_img = self.rotation(n_img, self.angle)

        n_img = np.expand_dims(n_img, axis=0)
        img_torch = torch.from_numpy(n_img.copy()).type(torch.float)

        ga = self.extract_age()

        return {'image': img_torch, 'ga': ga, 'key': self.key}

def center_slices(view):
    if view == 'L':
        ids = np.arange(start=40,stop=70)
    elif view == 'A':
        ids = np.arange(start=64,stop=94)
    else:
        ids = np.arange(start=48,stop=78)
    return ids

def data_augmentation(base_set, path, view, key, h, ids):
    transformations = {1: (True, None),
                       2: (False, -10), 3: (True, -10),
                       4: (False, -5), 5: (True, -5),
                       6: (False, 5), 7: (True, 5),
                       8: (False, 10), 9: (True, 10)}
    
    for x, specs in transformations.items():
        aug = img_dataset(path, view, key, size = h, horizontal_flip = specs[0], rotation_angle = specs[1])
        aug = Subset(aug,ids)
        base_set = torch.utils.data.ConcatDataset([base_set, aug])
    return base_set

def loader(tr_path, val_path, info_path, view, batch_size, h):
    train_id = os.listdir(tr_path)
    valid_id = os.listdir(val_path)

    ids = center_slices(view)
    train_set = img_dataset(root_dir = tr_path + train_id[0], view = view, key = train_id[0][:-4], info_file = info_path, size = h)
    train_set = Subset(train_set, ids)
    # train_set = data_augmentation(train_set, source_path+'train/'+train_id[0], view, 
    #                               train_id[0][:-4], h, ids)

    for idx, image in enumerate(train_id):
        if idx != 0:
            train_path = tr_path + image
            tr_set = img_dataset(root_dir = train_path, view = view, key = image[:-4], info_file = info_path, size = h)
            tr_set = Subset(tr_set,ids)
            # tr_set = data_augmentation(tr_set, train_path, view, image[:-4], h, ids)
            train_set = torch.utils.data.ConcatDataset([train_set, tr_set])

    valid_set = img_dataset(root_dir = val_path + valid_id[0], view = view, key = valid_id[0][:-4], info_file = info_path, size = h)
    valid_set = Subset(valid_set, ids)

    for idx, image in enumerate(valid_id):
        if idx != 0:
            valid_path = val_path + image
            val_set = img_dataset(root_dir = valid_path, view = view, key = image[:-4], info_file = info_path, size = h)
            val_set = Subset(val_set, ids)
            valid_set = torch.utils.data.ConcatDataset([valid_set, val_set])

    # Dataloaders generated from datasets 
    train_final = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=12, drop_last=True)
    valid_final = DataLoader(valid_set, shuffle=True, batch_size=batch_size, num_workers=12)
    return train_final, valid_final

def test_loader(source_path, view, batch_size, h, raw = False):
    test_id = os.listdir(source_path+'/')

    ids = center_slices(view)
    test_set = img_dataset(source_path+'/'+test_id[0], view, test_id[0][:-4], size = h, raw = raw)
    test_set = Subset(test_set,ids)

    test_data_list = []
    for idx,image in enumerate(test_id):
        if idx != 0:
            test_path = source_path + '/' + image
            ts_set = img_dataset(test_path,view, image[:-4], size = h, raw = raw)
            ts_set = Subset(ts_set,ids)
            test_set = torch.utils.data.ConcatDataset([test_set, ts_set])
            test_data_list.append(image[:-4])
            
    # Dataloaders generated from datasets 
    test_final = DataLoader(test_set, shuffle=False, batch_size=batch_size, num_workers=12)
    return test_final, test_data_list

def path_generator(args):
    # Define paths for obtaining dataset and saving models and results.
    folder_name = args.name+'_'+args.view

    tensor_path = args.path + 'Results/' + folder_name + '/history.txt'
    model_path = args.path + 'Results/' + folder_name + '/Saved_models/'
    image_path = args.path + 'Results/' + folder_name + '/Progress/'
    
    os.makedirs(args.path + 'Results/', exist_ok=True)

    if not os.path.exists(args.path + 'Results/' + folder_name):
        os.mkdir(args.path + 'Results/' + folder_name)
        os.mkdir(model_path)
        os.mkdir(image_path)
    print('Directories and paths are correctly initialized.')
    print('-'*25)
    return model_path, tensor_path, image_path

# def settings_parser():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--task',
#         dest='task',
#         choices=['Train', 'Validate'],
#         required=False,
#         default='Train',
#         help='''
#         Task to be performed.''')  
#     parser.add_argument('--VAE_model_type',
#         dest='VAE_model_type',
#         choices=['default', 'ga_VAE'],
#         default = 'default',
#         required=False,
#         help='''
#         Type of model to train. Available options:
#         "defalut" Default VAE using convolution blocks
#         "ga_VAE: VAE which includes GA as input''') 
#     parser.add_argument('--model_type',
#         dest='type',
#         choices=['default', 'bVAE'],
#         required=True,
#         help='''
#         Type of model to train. Available options:
#         "defalut" Default VAE using convolution blocks
#         "bVAE: VAE with disentanglement''')  
#     parser.add_argument('--model_view',
#         dest='view',
#         choices=['C', 'A', 'S'],
#         required=True,
#         help='''
#         The view of the image input for the model. Options:
#         "C" Coronal view
#         "A" Axial view
#         "S" Sagittal view''') 
#     parser.add_argument('--ga_method',
#         dest='ga_method',
#         choices=['multiplication', 'concat', 'concat_sample', 'ordinal_encoding', 'one_hot_encoding', 'boe'],
#         default = 'concat',
#         required=False,
#         help='''
#         Method to implement GA. Available options:
#         "multiplication", "concat"''') 
#     parser.add_argument('--gpu',
#         dest='gpu',
#         choices=['0', '1', '2'],
#         default='0',
#         required=False,
#         help='''
#         The GPU that will be used for training. Terminals have the following options:
#         Hanyang: 0, 1
#         Busan: 0, 1, 2
#         Sejong 0, 1, 2
#         Songpa 0, 1
#         Gangnam 0, 1
#         ''')
#     parser.add_argument('--epochs',
#         dest='epochs',
#         type=int,
#         default=2000,
#         required=False,
#         help='''
#         Number of epochs for training.
#         ''')    
#     parser.add_argument('--loss',
#         dest='loss',
#         default='L2',
#         choices=['L2', 'L1', 'SSIM', 'MS_SSIM'],
#         required=False,
#         help='''
#         Loss function for VAE:
#         L2 = Mean square error.
#         SSIM = Structural similarity index.
#         ''')
#     parser.add_argument('--batch',
#         dest='batch',
#         type=int,
#         default=32,
#         choices=[2**x for x in range(8)],
#         required=False,
#         help='''
#         Number of batch size.
#         ''') 
#     parser.add_argument('--z_dim',
#         dest='z_dim',
#         type=int,
#         default=512,
#         required=False,
#         help='''
#         z dimension.
#         ''')
#     parser.add_argument('--pretrained',
#         dest='pretrained',
#         type=str,
#         default=None,
#         choices=['base','refine'],
#         required=False,
#         help='''
#         If VAE model is pre-trained.
#         ''')
#     parser.add_argument('--pre_name',
#         dest='pre_n',
#         type=str,
#         default='Tlaloc',
#         required=False,
#         help='''
#         Name of pre-trained VAE model.
#         '''
#             )
#     parser.add_argument('--name',
#         dest='name',
#         type=str,
#         required=True,
#         help='''
#         Name for new VAE model.
#         '''
#             )
#     parser.add_argument('--slice_size',
#         dest='slice_size',
#         type=int,
#         default=158,
#         required=False,
#         help='''
#         Size of images from pre-processing (n x n).
#         ''')
#     parser.add_argument('--path',
#         dest = 'path',
#         type = str,
#         default = './',
#         required = False,
#         help='''
#         Path to the project directory
#         ''')
#     parser.add_argument('--training_folder',
#         dest = 'training_folder',
#         type = str,
#         default = 'TD_dataset/',
#         required = False,
#         help='''
#         Path to the project directory
#         ''')
#     parser.add_argument(
#         '-ga_n',
#         '--GA_encoding_dimensions', 
#         dest='ga_n',
#         type=int,
#         default=100,
#         required=False,
#         help='''
#         Size of vector for ga representation.
#         ''')
#     parser.add_argument(
#         '-raw',
#         '--raw_data', 
#         dest='raw',
#         action='store_true',
#         required=False,
#         help='''
#         Training or testing on raw data.
#         ''')
#     parser.add_argument(
#         '-th',
#         '--threshold', 
#         dest='th',
#         type=int,
#         default=99,
#         help='''
#         Treshold for the mask.
#         ''')
#     parser.add_argument(
#         '-cGAN',
#         '--conditional_GAN', 
#         dest='cGAN',
#         action='store_true',
#         required=False,
#         help='''
#         BOE implemented to the GD as a cGAN.
#         ''')
#     parser.add_argument(
#         '-cGAN_s',
#         '--conditional_GAN_size', 
#         dest='cGAN',
#         action='store_true',
#         required=False,
#         help='''
#         BOE implemented to the GD as a cGAN.
#         ''')
#     parser.add_argument('--beta_kl',
#         dest='beta_kl',
#         type=float,
#         default=None,
#         required=False,
#         help='''
#         The value of the beta KL parameter.
#         ''')
#     parser.add_argument('--beta_rec',
#         dest='beta_rec',
#         type=float,
#         default=None,
#         required=False,
#         help='''
#         The value of the beta rec parameter.
#         ''')
#     parser.add_argument('--beta_neg',
#         dest='beta_kl',
#         type=float,
#         default=None,
#         required=False,
#         help='''
#         The value of the beta neg parameter.
#         ''')
#     parser.add_argument('--BOE_type',
#         dest='BOE_type',
#         default='BOE',
#         choices=['BOE', 'EBOE', 'inv_BOE', 'inv_inv_BOE'],
#         required=False,
#         help='''
#         BOE type.
#         ''')
    
#     return parser