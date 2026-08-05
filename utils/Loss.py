
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import WassersteinDistance,CubicalComplex
import matplotlib.pyplot as plt
import gudhi as gd  
import gudhi.representations
#from skimage.feature import local_binary_pattern 
#import gudhi as gd
#from gudhi.wasserstein import wasserstein_distance
#import gudhi as gd
import numpy as np
import torch

def batch_persistence_images(pi_s, pi_t, pimg):
    """
    Compute combined (H0+H1) persistence images for a batch.
    
    Args:
        pi_s: list of PDs for student, shape [B][2], pi_s[b][0]=H0, pi_s[b][1]=H1
        pi_t: same as pi_s but for teacher
        pimg: gudhi.representations.PersistenceImage() instance

    Returns:
        pimg_s: torch.Tensor [B, H, W]
        pimg_t: torch.Tensor [B, H, W]
    """
    batch_size = len(pi_s)
    imgs_s, imgs_t = [], []

    for b in range(batch_size):
        # Student combine H0 + H1
        combined_s = torch.cat([pi_s[b][0][1], pi_s[b][1][1]], dim=0)  # (n_s, 2)
        combined_s = combined_s.detach().cpu().numpy()
        img_s = pimg.fit_transform([combined_s])[0]  # (H, W)
        imgs_s.append(img_s)

        # Teacher combine H0 + H1
        combined_t = torch.cat([pi_t[b][0][1], pi_t[b][1][1]], dim=0)  # (n_t, 2)
        combined_t = combined_t.detach().cpu().numpy()
        img_t = pimg.fit_transform([combined_t])[0]  # (H, W)
        imgs_t.append(img_t)

    # Stack and convert to torch tensor
    pimg_s = torch.tensor(np.stack(imgs_s)).float()  # [B, H, W]
    pimg_t = torch.tensor(np.stack(imgs_t)).float()  # [B, H, W]
    # Reshape back into images
    H, W = 32, 32
    pimg_s = pimg_s.view(batch_size, H, W)
    pimg_t = pimg_t.view(batch_size, H, W)
    return pimg_s, pimg_t

def visualize_segmentation_and_persistence(s_out, pi_s,wloss,batch_ids=[0,1]):
    """
    s_out: Student segmentation outputs (B, 1, H, W)
    pi_s: Student persistence diagrams, list of length B, each element is a list of diagrams for different features
    """
    # batch_id'ler
    features_list = [0, 1]  # H0=0, H1=1
    pd = 1  # persistence diagram seçeneği

    fig, axes = plt.subplots(len(batch_ids), 2, figsize=(6, 6))

    for i, batch in enumerate(batch_ids):
        # ---- Sol tarafta segmentasyon output ----
        axes[i, 0].imshow(s_out[batch].detach().cpu().numpy().squeeze(), cmap="gray")
        axes[i, 0].set_title(f"Segmentation Output (Batch {batch})")
        axes[i, 0].axis("off")

        # ---- Sağ tarafta persistence diagram ----
        for features, color in zip(features_list, ["blue", "red"]):
            births = pi_s[batch][features][pd][:, 0]
            deaths = pi_s[batch][features][pd][:, 1]
            axes[i, 1].scatter(births.detach().cpu().numpy(), deaths.detach().cpu().numpy(), s=20, label=f"H{features}", alpha=0.7, color=color)

        # y=x diyagonal çizgisi
        all_vals = [pi_s[batch][f][pd].detach().cpu().numpy() for f in features_list]
        min_val = min([arr.min() for arr in all_vals])
        max_val = max([arr.max() for arr in all_vals])
        axes[i, 1].plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5)

        axes[i, 1].set_title(f"Persistence Diagram (Batch {batch})")
        axes[0, 1].set_title(f"w-loss {wloss})")
        axes[i, 1].legend()
        axes[i, 1].set_xlabel("Birth")
        axes[i, 1].set_ylabel("Death")

    plt.tight_layout()
    plt.show()


class DINOLoss(nn.Module):
    def __init__(self, out_dim=256, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.topo_loss = Topological_Loss()

    def forward(self, student_outputs, teacher_outputs, label,teacher_temp):

        total_loss, n_loss_terms = 0.0, 0
        for t_out in teacher_outputs:  # loop over teacher views (2)
            for s_out in student_outputs:  # loop over student views (2)
                # Compute topo_loss per image
                topo_loss = self.topo_loss(t_out,s_out,label[0])
                total_loss += topo_loss
                n_loss_terms += 1

        return total_loss / n_loss_terms


class Topological_Loss(torch.nn.Module):

    def __init__(self, lam=0.1, topology_image_size=32):
        super().__init__()
        self.lam                = lam
        self.topology_image_size = topology_image_size
        #self.vr                 = VietorisRipsComplex(dim=self.dimension)
        self.cubicalcomplex     = CubicalComplex()
        self.wloss              = WassersteinDistance(p=2)
        self.sigmoid_f          = nn.Sigmoid()
        self.avgpool            = nn.AvgPool2d(2,2)

    def _downsample_to_topology_size(self, tensor):
        """Repeatedly halve a square tensor until topology_image_size is reached."""
        height, width = tensor.shape[-2:]
        if height != width:
            raise ValueError(
                f"Topological_Loss expects square inputs, got {height}x{width}."
            )
        if self.topology_image_size <= 0 or self.topology_image_size > height:
            raise ValueError(
                f"topology_image_size must be in [1, {height}], "
                f"got {self.topology_image_size}."
            )

        while height > self.topology_image_size:
            if height % 2 != 0 or height // 2 < self.topology_image_size:
                raise ValueError(
                    f"Cannot reach {self.topology_image_size}x{self.topology_image_size} "
                    f"from {tensor.shape[-2]}x{tensor.shape[-1]} using 2x2 AvgPool."
                )
            tensor = self.avgpool(tensor)
            height, width = tensor.shape[-2:]
        return tensor
  
    def forward(self, model_output,labels):

        totalloss             = 0
        # Work on probabilities, then reduce the spatial resolution for a
        # substantially cheaper cubical-complex computation.
        model_output_r        = self.sigmoid_f(model_output)
        model_output_r        = self._downsample_to_topology_size(model_output_r)
        labels_r              = self._downsample_to_topology_size(labels)
        predictions           = torch.squeeze(model_output_r,dim=1) 
        masks                 = torch.squeeze(labels_r,dim=1)
        pi_pred               = self.cubicalcomplex(predictions)
        pi_mask               = self.cubicalcomplex(masks)
        
        for i in range(predictions.shape[0]):

            topo_loss   = self.wloss(pi_mask[i],pi_pred[i])             
            totalloss   +=topo_loss
        loss             = self.lam * totalloss/predictions.shape[0]
        return loss, pi_pred, pi_mask

class Topological_Loss_distillation(torch.nn.Module):

    def __init__(self, lam=0.1):
        super().__init__()
        self.lam                = lam
        #self.vr                 = VietorisRipsComplex(dim=self.dimension)
        self.cubicalcomplex     = CubicalComplex()
        self.wloss              = WassersteinDistance(p=2)
        self.sigmoid_f          = nn.Sigmoid()

    def forward(self, t_out, s_out,label):

        totalloss             = 0
        t_out                 = torch.squeeze(self.sigmoid_f(t_out),dim=1)
        s_out                 = torch.squeeze(self.sigmoid_f(s_out),dim=1)
        pi_t                  = self.cubicalcomplex(t_out)
        pi_s                  = self.cubicalcomplex(s_out)
        # Compute topological loss for each image in the batch
        pimg = gd.representations.PersistenceImage(bandwidth=0.05, resolution=(32,32), weight=lambda x: 1.0)

        pimg_s, pimg_t = batch_persistence_images(pi_s, pi_t, pimg)

        for i in range(t_out.shape[0]):
            topo_loss   = self.wloss(pi_s[i],pi_t[i]) 
            totalloss   +=topo_loss
        loss             = self.lam * totalloss/t_out.shape[0]
        return loss

class Dice_CE_Loss():
    def __init__(self):

#        self.batch,self.h,self.w,self.n_class = inputs.shape

        self.sigmoid_f     = nn.Sigmoid()
        self.softmax       = nn.Softmax(dim=-1)
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.bcewithlogic = nn.BCEWithLogitsLoss(reduction="mean")
    
    def Dice_Loss(self,input,target):

        smooth          = 1
        input           = self.sigmoid_f(torch.flatten(input=input))
        target          = torch.flatten(input=target)
        intersection    = (input * target).sum()
        dice_loss       = 1- (2.*intersection + smooth )/(input.sum() + target.sum() + smooth)
        return dice_loss

    def BCE_loss(self,input,target):
        input           = torch.flatten(input=input)
        target          = torch.flatten(input=target)
        sigmoid_f       = nn.Sigmoid()
        sigmoid_input   = sigmoid_f(input)
        #B_Cross_Entropy = F.binary_cross_entropy(sigmoid_input,target)
        entropy_with_logic = self.bcewithlogic(input,target)
        return entropy_with_logic

    def Dice_BCE_Loss(self,input,target):
        return self.Dice_Loss(input,target) + self.BCE_loss(input,target) 
    
    
    # Manuel cross entropy loss 
    def softmax_manuel(self,input):
        return (torch.exp(input).t() / torch.sum(torch.exp(input),dim=1)).t()

    def CE_loss_manuel(self, input,target):

        last_dim = torch.tensor(input.shape[:-1])
        last_dim = torch.prod(last_dim)
        input    = input.reshape(last_dim,-1)      
        target   = target.view(last_dim,-1)     #    should be converted one hot previously

        return torch.mean(-torch.sum(torch.log(self.softmax_manuel(input)) * (target),dim=1))


    # CE loss 
    def CE_loss(self,input,target):
        cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        last_dim = torch.tensor(input.shape[:-1])
        last_dim = torch.prod(last_dim)
        input    = input.reshape(last_dim,-1)
        target   = target.reshape(last_dim).long         #  it will be converted one hot encode in nn.CrossEnt 

        return cross_entropy(input,target)
