# Code written by @Sungmin, @simonamador, and @GuillermoTafoya

from train_framework import Trainer
from utils.util import *
import os, torch, argparse

#python Src/main.py --task Training --model_type CCVAEGAN --model_view S --training_path Dataset/0_mixed_site_cohort/TD_train/ --validation_path Dataset/0_mixed_site_cohort/TD_test/ --info_path Dataset/TD_data_proceed.csv --gpu 0 --slice_size 158 --batch 256 --z_dim 512 --beta_rec 1 --beta_kl 10 --name CCVAEGAN_s158_b256_z512_ga100_br1_bkl10

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # Experiment setting
    parser.add_argument('--task', dest='task', choices=['Training', 'Evaluation'], required=False, default='Training', help='''Task to be performed.''')  
    parser.add_argument('--model_type', dest='model_type', choices=['VAEGAN', 'GAVAEGAN', 'CVAEGAN', 'cycleVAEGAN', 'cycleGAVAEGAN', 'CCVAEGAN'], default = 'CCVAEGAN', required=False, 
                        help='''Type of model to train. Available options:
                            "VAEGAN" Default VAE using convolution blocks
                            "GAVAEGAN" VAEGAN with GA feeding
                            "CVAEGAN" (GA) Conditioned VAEGAN
                            "cycleVAEGAN" Cycle VAEGAN
                            "cycleGAVAEGAN" Cycle VAEGAN with GA feeding
                            "CCVAEGAN" Conditional Cyclic variational autoencoding generative adversarial network''')  
    parser.add_argument('--model_view', dest='view', choices=['C', 'A', 'S'], required=True, 
                        help='''The view of the image input for the model. Options:
                        "C" Coronal view
                        "A" Axial view
                        "S" Sagittal view''') 
    parser.add_argument('--gpu', dest='gpu', choices=['0', '1', '2'], default='0', required=False)
    parser.add_argument('--epochs', dest='epochs', type=int, default=10000, required=False)    
    parser.add_argument('--loss', dest='loss', default='L2', choices=['L2', 'L1', 'SSIM', 'MS_SSIM'], required=False)
    parser.add_argument('--batch', dest='batch', type=int, default=256, required=False) 
    parser.add_argument('--z_dim', dest='z_dim', type=int, default=512, required=False)
    parser.add_argument('--path', dest = 'path', type = str, default = './', required = False, help='''Path to the project directory''')
    parser.add_argument('--name', dest='name', type=str, required=True, help='''Name for experiemnt''')
    parser.add_argument('--training_path', dest = 'tr_path', type = str, required = True, help='''Path to the training data''')
    parser.add_argument('--validation_path', dest = 'val_path', type = str, required = True, help='''Path to the validation data''')
    parser.add_argument('--info_path', dest = 'info_path', type = str, required = True, help='''Path to the information CSV file''')

    # Hyperparameter options
    parser.add_argument('--ga_method', dest='ga_method', choices=['multiplication', 'concat', 'concat_sample', 'ordinal_encoding', 'one_hot_encoding', 'boe'],
                        default = 'boe', required=False, help='''Method to feed GA''')
    parser.add_argument('--slice_size', dest='slice_size', type=int, default=158, required=False, help='''Size of images from pre-processing (n x n).''')
    parser.add_argument('-ga_n', '--GA_encoding_dimensions', dest='ga_n', type=int, default=100, required=False, help='''Size of vector for ga representation.''')
    parser.add_argument('--beta_kl', dest='beta_kl', type=float, default=10, required=False, help='''The value of the beta KL parameter.''')
    parser.add_argument('--beta_rec', dest='beta_rec', type=float, default=1, required=False, help='''The value of the beta rec parameter.''')
    parser.add_argument('--loss_scale', dest='loss_scale', type=float, default=1, required=False, help='''Scaling factor for entire loss''')

    # Evaluation options
    parser.add_argument('--test_data_path', dest = 'test_data_path', type = str, required = False, help='''Path to the test data''')
    parser.add_argument('--test_info_path', dest = 'test_info_path', type = str, required = False, help='''Path to the information CSV file''')
    parser.add_argument('--test_output_path', dest = 'test_output_path', type = str, required = False, help='''Path for test outcomes''')
    parser.add_argument('--test_outcome_path', dest = 'test_outcome_path', type = str, required = False, help='''Path for test outcome scores in *.csv ''')
    parser.add_argument('--viz_option', dest = 'test_viz_option', type = str, required = False, choices = ['all', 'center', 'no'], default='no', help='''Anomaly map save option''')
    parser.add_argument('--vmax', dest = 'vmax', required = False, default = [0.4, 0.08], help='''Max value for anomaly map visualization''')
    parser.add_argument('--save_npy', dest = 'save_npy', type = bool, required = False, default=False, help='''Flag to save generation in numpy array (for external brain age model)''')
       
    # Obtain all configs from parser
    args = parser.parse_args()

    # Establish CUDA GPU
    torch.autograd.set_detect_anomaly(True)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print(f'{torch.cuda.is_available()=}')
    print(f'{torch.cuda.device_count()=}')

    if args.task == "Training":
        # Obtain all paths needed for training/validation from config
        model_path, tensor_path, image_path = path_generator(args)

        # Convert args to a dictionary and add device and paths dynamically
        parameters = vars(args).copy()
        parameters['device'] = device.type  
        parameters['model_path'] = model_path
        parameters['tensor_path'] = tensor_path
        parameters['image_path'] = image_path

        trainer = Trainer(parameters)
        trainer.train(args.epochs, args.loss)
    elif args.task == "Evaluation":
        print("Evaluation")

        test_data_path = args.test_data_path
        test_info_path = args.test_info_path
        test_output_path = args.test_output_path
        
        tester=Tester(test_data_path, test_output_path, test_info_path, vmax=args.vmax)
        val_sim_idx, val_DIS_idx, Subject_ID, GT_GA, Pred_GA, Gen_GA, anomaly_score_L1, anomaly_score_L2, anomaly_score_SSIM, anomaly_score_MSSIM = tester.test()

        df = pd.DataFrame({
            'Subject_ID': Subject_ID,
            'GT_GA': np.squeeze(GT_GA),
            'Pred_GA': np.squeeze(Pred_GA),
            'Gen_GA': np.squeeze(Gen_GA),
            'anomaly_score_L1': np.squeeze(anomaly_score_L1),
            'anomaly_score_L2': np.squeeze(anomaly_score_L2),
            'anomaly_score_SSIM': np.squeeze(anomaly_score_SSIM),
            'anomaly_score_MSSIM': np.squeeze(anomaly_score_MSSIM)
        })
        df.to_csv(test_outcome_path, index=False)

