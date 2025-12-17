import numpy as np
import torch
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
#from .patch_alignment_loss import PatchwiseContentLoss
import torch.nn.functional as F
import torch.nn as nn
from .PatchGCL import GNNLoss
import os
class M2PLGANModel(BaseModel):
    """ .
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss: GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',type=util.str2bool, nargs='?', const=True, default=False, help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',type=util.str2bool, nargs='?', const=True, default=False, help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")
        
        parser.set_defaults(pool_size=0)  # no image pooling
        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(nce_idt=False, lambda_NCE=10.0, flip_equivariance=False,n_epochs=100, n_epochs_decay=10)
        else:
            raise ValueError(opt.CUT_mode)
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'NCE','G','D','D_real','D_fake', 'CACM', 'LDAM','GBCLM']
        self.visual_names = ['fake_B1']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]
        self.opt=opt
        print('opt.lambda_NCE:',opt.lambda_NCE)
        if opt.nce_idt and self.isTrain:
            self.loss_names += ['identity_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
        
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = PatchNCELoss(opt).to(self.device)
            self.GNNLoss = GNNLoss(opt,use_mlp=True).to(self.device)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)


    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        #bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.current_epoch=data['current_epoch']
        print('self.current_epoch:',self.current_epoch)
        
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
    
    def optimize_parameters(self):
        # forward
        self.forward()
        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()
        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()
        
    def set_input(self, input):
        if 'current_epoch' in input:
            self.current_epoch = input['current_epoch']
        if 'current_iter' in input:
            self.current_iter = input['current_iter']
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
         
        self.real_A1 = input['A1' if AtoB else 'B1'].to(self.device)
        self.real_B1 = input['B1' if AtoB else 'A1'].to(self.device)
        
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        if  self.opt.phase == "val" and self.opt.ablation_fake_dataroot is not None:
            
            self.ablation_fake_img = input['fake_img'].to(self.device)

    def multi_level_feature(self):
        
        feat_fake_B1 = self.netG(self.ablation_fake_img, self.nce_layers, encode_only=True)
        
        feat_real_B1 = self.netG(self.real_B1, self.nce_layers, encode_only=True)
        
        return feat_fake_B1, feat_real_B1
    
    def forward(self):
        # self.netG.print()
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        #self.real = torch.cat((self.real_A1, self.real_A2, self.real_A3, self.real_B1, self.real_B2, self.real_B3), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)

            if self.flipped_for_equivariance:
                self.real_A1 = torch.flip(self.real_A1, [3]) 
        self.fake_B1 = self.netG(self.real_A1, layers=[])

        if self.opt.nce_idt:
            self.idt_B = self.netG(self.real_B1)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        
        # Fake; stop backprop to the generator by detaching fake_B
        fake_1 = self.fake_B1.detach()
        pred_fake1 = self.netD(fake_1)
        self.loss_D_fake = self.criterionGAN(pred_fake1, False).mean()

        # Real
        pred_real1 = self.netD(self.real_B1)
        self.loss_D_real = self.criterionGAN(pred_real1, True).mean()
        self.loss_D = ( + self.loss_D_fake  + self.loss_D_real ) * 0.5

        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        loss_NCE_all = 0.0

        feat_real_A1 = self.netG(self.real_A1, self.nce_layers, encode_only=True)
        feat_fake_B1 = self.netG(self.fake_B1, self.nce_layers, encode_only=True)
        feat_real_B1 = self.netG(self.real_B1, self.nce_layers, encode_only=True)
        
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake1 = self.netD(self.fake_B1)
            self.loss_G_GAN = self.criterionGAN(pred_fake1, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(feat_real_A1, feat_fake_B1, self.netF, self.nce_layers)
        else:
            self.loss_NCE =  0.0
        
        
        if self.opt.CACM > 0.0:
            self.loss_CACM =self.calculate_ss_loss(feat_real_B1,feat_fake_B1) * self.opt.CACM
        else:
            self.loss_CACM = 0.0

            
        if self.opt.LDAM > 0.0:
            if self.opt.pixelwpiexl:
                self.loss_LDAM = self.calculate_mmd_loss(feat_fake_B1,feat_real_B1,pixelwpiexl=True)  * self.opt.LDAM
            elif self.opt.patchwpatch:
                self.loss_LDAM = self.calculate_mmd_loss(feat_fake_B1,feat_real_B1,patchwpatch=True)  * self.opt.LDAM
            else: 
                raise KeyError
        else:
            self.loss_LDAM = 0.0

        if self.opt.GBCLM > 0.0:
            
            if self.opt.LDAM > 0.0:
                pass
            else:
                self.feats_y, sample_ids = self.netF(feat_fake_B1, self.opt.num_patches, None)
                self.feats_x, _ = self.netF(feat_real_B1, self.opt.num_patches, sample_ids)
           
            self.feats_y, sample_ids = self.netF(feat_fake_B1, self.opt.num_patches, None)
            self.feats_x, _ = self.netF(feat_real_B1, self.opt.num_patches, sample_ids)
            self.loss_GBCLM = self.GNNLoss(self.feats_y, self.feats_x, num_patches=self.opt.num_patches) * self.opt.GBCLM
            
            
        else:
            self.loss_GBCLM = 0.0
    

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_identity_Y = self.calculate_identity_loss(self.real_B1, self.real_B1) 
        else:
            self.loss_identity_Y  =0.0

        loss_NCE_all +=   self.loss_NCE 
        self.loss_G = self.loss_G_GAN + loss_NCE_all + self.loss_CACM + self.loss_LDAM + self.loss_GBCLM
        return self.loss_G


    def calculate_NCE_loss(self, feat_src, feat_tgt, netF,nce_layers):
        n_layers = len(feat_src)
        feat_q = feat_tgt

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]
            
        feat_k = feat_src
        feat_k_pool, sample_ids = netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = netF(feat_q, self.opt.num_patches, sample_ids)
        
        total_nce_loss = 0.0
        for f_q, f_k in zip(feat_q_pool, feat_k_pool):
            
            loss = self.criterionNCE(f_q, f_k) 
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers
    
    def calculate_ss_loss(self, feat_real_B, feat_fake_B):
        total_ss_loss = 0.0
        for idx, (real_feat, fake_feat) in enumerate(zip(feat_real_B, feat_fake_B)):
            if real_feat.shape[0]== 1:  # 单张图
                matrix_src = self.calculate_patch_similarity(real_feat.squeeze(0), patch_size=self.opt.patchsize, stride=self.opt.patchsize).to(self.device)
                matrix_tgt = self.calculate_patch_similarity(fake_feat.squeeze(0), patch_size=self.opt.patchsize, stride=self.opt.patchsize).to(self.device)
                layer_loss = F.l1_loss(matrix_src, matrix_tgt)
                total_ss_loss += layer_loss
            else:
                raise KeyError("Invalid feature shape")

        return total_ss_loss    


    
    def calculate_patch_similarity(self, image, patch_size=64, stride=64):
        """
        image: torch.Tensor, shape (C, H, W)
        return: similarity matrix of shape (n_patches, n_patches)
        """
        
        C, H, W = image.shape
        patches = image.unfold(1, patch_size, stride).unfold(2, patch_size, stride)  # (C, n_h, n_w, patch_size, patch_size)
        patches = patches.permute(1, 2, 0, 3, 4)  # (n_h, n_w, C, patch_size, patch_size)
        patches = patches.contiguous().view(-1, C * patch_size * patch_size)  # (n_patches, C * patch_area)
        patches = F.normalize(patches, dim=1)  # normalize for cosine similarity
        similarity_matrix = torch.matmul(patches, patches.T)  # (n_patches, n_patches)
        
        return similarity_matrix
  
    def compute_kernel(self,x, y, kernel='rbf', sigma=1.0):
       
        x_size = x.size(0)
        y_size = y.size(0)

        xx = x.unsqueeze(1).expand(x_size, y_size, -1)
        yy = y.unsqueeze(0).expand(x_size, y_size, -1)
        dist = (xx - yy).pow(2).sum(2)

        if kernel == 'rbf':
            K = torch.exp(-dist / (2 * sigma ** 2))
        else:
            raise NotImplementedError("Only RBF kernel implemented.")

        return K
    
    def calculate_loss(self, x, y, sigma=1.0):
        
        K_xx = self.compute_kernel(x, x, sigma=sigma)
        K_yy = self.compute_kernel(y, y, sigma=sigma)
        K_xy = self.compute_kernel(x, y, sigma=sigma)

        mmd = K_xx.mean() + K_yy.mean() - 2 * K_xy.mean()
        return mmd 
    
    def extract_patches(self, feat, patch_size, stride):
               
        patches = feat.unfold(2, patch_size, stride).unfold(3, patch_size, stride)  # [1, C, n_h, n_w, patch_size, patch_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5)  # [1, n_h, n_w, C, patch_size, patch_size]
        patches = patches.contiguous().view(-1, feat.size(1), patch_size, patch_size) # [n_patches, C, p, p]
        
        pooled = F.adaptive_avg_pool2d(patches, output_size=1)  # [n_patches, C, 1, 1]
        
        #pooled = F.adaptive_max_pool2d(patches, output_size=1)  # [n_patches, C, 1, 1]
        
        pooled = pooled.view(pooled.size(0), -1)  # [n_patches, C]
        patches = F.normalize(pooled, dim=1)
        
        
        return patches
    
    def calculate_mmd_loss(self, feats_x, feats_y, sigma=1.0, pixelwpiexl=False,patchwpatch=False):
        """
        feats_x, feats_y: list of feature maps, each of shape [1, C, H, W]
        return: scalar MMD loss
        """
        total_loss = 0.0
        weights = [1.0, 0.8, 0.5, 0.3, 0.3]
        
        if pixelwpiexl:
           
            self.feats_y, sample_ids = self.netF(feats_y, self.opt.num_patches, None)
            self.feats_x, _ = self.netF(feats_x, self.opt.num_patches, sample_ids)
                                
            for idx, (x, y) in enumerate(zip(self.feats_x, self.feats_y)):
                # compute MMD on flattened features
                mmd = self.calculate_loss(x, y, sigma=sigma)
                total_loss += weights[idx] * mmd

        elif patchwpatch:
            self.feats_y_list = []
            self.feats_x_list = []
            
            for idx, (fx, fy) in enumerate(zip(feats_x, feats_y)):
                x_patches = self.extract_patches(fx, patch_size=16, stride=16)
                y_patches = self.extract_patches(fy, patch_size=16, stride=16)

                self.feats_y_list.append(x_patches)
                self.feats_x_list.append(y_patches)
                # 计算MMD
                mmd = self.calculate_loss(x_patches, y_patches, sigma=sigma)
                total_loss +=  mmd

        else:
            raise KeyError("Invalid.")
        return total_loss
    