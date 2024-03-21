import torch
import torch.nn as nn
from ldm.modules.diffusionmodules.util import FourierEmbedder

class CombinedPositionNet(nn.Module):
    def __init__(self, in_dim, max_persons_per_image, out_dim, fourier_freqs=8):
        super().__init__()
        self.in_dim = in_dim
        self.max_persons_per_image = max_persons_per_image
        self.out_dim = out_dim

        # Fourier embedder for both boxes (xyxy) and keypoints (xy)
        self.fourier_embedder = FourierEmbedder(num_freqs=fourier_freqs)
        self.box_position_dim = fourier_freqs * 2 * 4  # for boxes: 2 for sin & cos, 4 for xyxy
        self.kp_position_dim = fourier_freqs * 2 * 2  # for keypoints: 2 for sin & cos, 2 for xy

        self.person_embeddings = torch.nn.Parameter(torch.zeros([max_persons_per_image, out_dim]))
        self.keypoint_embeddings = torch.nn.Parameter(torch.zeros([17, out_dim]))  # Adjust 17 to your number of keypoints if different

        # Adjust input dimensions to include both box and keypoint positional embeddings
        total_position_dim = self.box_position_dim + self.kp_position_dim
        self.linears = nn.Sequential(
            nn.Linear(in_dim + total_position_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, out_dim),
        )

        self.null_positive_feature = torch.nn.Parameter(torch.zeros([self.in_dim]))
        self.null_box_feature = torch.nn.Parameter(torch.zeros([self.box_position_dim]))
        self.null_kp_feature = torch.nn.Parameter(torch.zeros([self.kp_position_dim]))

    def forward(self, boxes, points, masks, positive_embeddings):
        B, N, _ = boxes.shape
        masks = masks.unsqueeze(-1)

        # Embed bounding boxes and keypoints using Fourier embeddings
        box_embedding = self.fourier_embedder(boxes)  # B*N*4 --> B*N*C
        kp_embedding = self.fourier_embedder(points)   # Adjust dimensions if necessary

        # Null embeddings for padding
        positive_null = self.null_positive_feature.view(1, 1, -1)
        box_null = self.null_box_feature.view(1, 1, -1)
        kp_null = self.null_kp_feature.view(1, 1, -1)

        # Replace padding with null embeddings
        positive_embeddings = positive_embeddings * masks + (1 - masks) * positive_null
        box_embedding = box_embedding * masks + (1 - masks) * box_null
        kp_embedding = kp_embedding * masks + (1 - masks) * kp_null

        # Concatenate all embeddings
        combined_embedding = torch.cat([positive_embeddings, box_embedding, kp_embedding], dim=-1)
        objs = self.linears(combined_embedding)
        assert objs.shape == torch.Size([B, N, self.out_dim])
        return objs
