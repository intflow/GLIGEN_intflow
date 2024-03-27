import torch
import torch.nn as nn
from ldm.modules.attention import BasicTransformerBlock
from ldm.modules.diffusionmodules.util import checkpoint, FourierEmbedder
import torch.nn.functional as F



class PositionNet(nn.Module):
    def __init__(self, max_boxes_per_data, in_dim, out_dim, num_kp=9, fourier_freqs=8):
        super().__init__()
        self.max_boxes_per_data = max_boxes_per_data
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_kp = num_kp

        # Adjust the position_dim for rbbox: cx, cy, w, h, angle
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.position_dim = fourier_freqs*2*5  # Updated for rbbox (cx, cy, w, h, angle)
        self.position_dim_kp_unit = fourier_freqs*2*2  # Keypoints remain unchanged
        self.position_dim_kp = self.position_dim_kp_unit * self.num_kp
        self.position_dim_input = self.position_dim + self.position_dim_kp

        self.linears = nn.Sequential(
            nn.Linear(self.in_dim + self.position_dim_input, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, self.out_dim),
        )

        # Null embeddings remain the same
        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_position_feature = torch.nn.Parameter(torch.zeros([self.position_dim]))
        self.null_kp_feature = torch.nn.Parameter(torch.zeros([self.position_dim_kp_unit]))

    def forward(self, rbboxes, masks, positive_embeddings, points, kp_masks):
        B, N, _ = rbboxes.shape
        masks = masks.unsqueeze(-1)
        kp_masks = kp_masks.unsqueeze(-1)

        # Embed rbbox positions, including angles
        rbbox_embedding = self.fourier_embedder(rbboxes)  # Updated for rbbox

        # Apply null embeddings for padding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        rbbox_null = self.null_position_feature.view(1, 1, -1)
        
        positive_embeddings = positive_embeddings*masks + (1-masks)*positive_null
        rbbox_embedding = rbbox_embedding*masks + (1-masks)*rbbox_null

        # Keypoint embeddings remain similar
        kp_embedding_all = self.embed_keypoints(points, kp_masks)

        # Concatenate all embeddings and pass through the network
        objs = self.linears(torch.cat([positive_embeddings, rbbox_embedding, kp_embedding_all], dim=-1))
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs
    
    def embed_keypoints(self, points, kp_masks):
        B, N, _ = points.shape
        kp_embedding_all = torch.zeros([B, N, self.position_dim_kp_unit, self.num_kp]).to(points.device)
        for k in range(self.num_kp):
            kp_embedding = self.fourier_embedder(points[:, :, 2*k:2*k+2])  # Embed each keypoint
            kp_null = self.null_kp_feature.view(1, 1, -1)
            kp_embedding_all[:, :, :, k] = kp_embedding*kp_masks[:, :, k, :] + (1-kp_masks[:, :, k, :])*kp_null
        kp_embedding_all = torch.reshape(kp_embedding_all, (B, N, -1))
        return kp_embedding_all
