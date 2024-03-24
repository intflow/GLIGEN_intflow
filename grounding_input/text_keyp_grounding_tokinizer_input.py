import os
import torch as th

class GroundingNetInput:
    def __init__(self, num_kp=9):
        self.set = False
        self.num_kp = num_kp

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        This method now also processes a keypoints tensor expected in the batch.
        The keypoints are associated with each bounding box, and there are 9 keypoints per box.
        """
        self.set = True

        boxes = batch['boxes']
        masks = batch['masks']
        positive_embeddings = batch["text_embeddings"]
        # Extracting keypoints from the batch. Expected shape: [batch_size, max_boxes, 9, 2]
        # Each keypoint is represented by a 2D coordinate (x, y).
        points = batch['points'] 
        kp_masks = batch['kp_masks']

        self.batch, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        return {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings, "points":points, "kp_masks":kp_masks}

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        This method now also generates a null tensor for keypoints, alongside boxes, masks, and positive embeddings.
        """
        assert self.set, "Not set yet, cannot call this function"

        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes = th.zeros(batch, self.max_box, 4).type(dtype).to(device)
        masks = th.zeros(batch, self.max_box).type(dtype).to(device)
        positive_embeddings = th.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device)

        points = th.zeros(batch, self.max_box, 2*self.num_kp).to(device) 
        kp_masks = th.zeros(batch, self.max_box, self.num_kp).to(device) 

        return {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings, "points": points, "kp_masks": kp_masks}
