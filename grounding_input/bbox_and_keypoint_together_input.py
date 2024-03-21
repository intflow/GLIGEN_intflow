import os
import torch as th

class GroundingNetInput:
    def __init__(self):
        self.set = False

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        This function processes the batch to prepare the input for the grounding tokenizer,
        combining both bounding boxes and keypoints.
        """

        self.set = True

        # Assuming bounding boxes and keypoints are part of the same batch dict
        boxes = batch['bboxes']
        points = batch['points']
        masks = batch['masks']  # Assuming the mask applies universally, adjust if separate masks are needed
        positive_embeddings = batch.get("text_embeddings", None)  # Text embeddings are optional

        # Adjust dimensions and prepare combined tensors if needed
        self.batch_size = boxes.shape[0]
        self.device = boxes.device
        self.dtype = boxes.dtype

        # Combine or concatenate bounding boxes and keypoints here
        # Example: concatenated along a new dimension or managed separately as per model requirements
        combined_input = {
            "boxes": boxes,
            "points": points,
            "masks": masks
        }

        # Include text embeddings if available
        if positive_embeddings is not None:
            combined_input["positive_embeddings"] = positive_embeddings

        return combined_input

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference,
        defines the null input for the grounding tokenizer combining both boxes and points.
        """

        assert self.set, "not set yet, cannot call this function"
        batch = self.batch_size if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        # Adjust dimensions as per your combined data
        # Example dimensions used here; adjust according to actual data structure
        max_box = self.max_box if hasattr(self, 'max_box') else 10
        max_persons_per_image = self.max_persons_per_image if hasattr(self, 'max_persons_per_image') else 10
        in_dim = self.in_dim if hasattr(self, 'in_dim') else 768  # Example embedding dimension

        boxes = th.zeros(batch, max_box, 4).type(dtype).to(device)
        points = th.zeros(batch, max_persons_per_image*9, 2).type(dtype).to(device)
        masks = th.zeros(batch, max_box).type(dtype).to(device)  # Adjust if separate masks for boxes and points are needed
        positive_embeddings = th.zeros(batch, max_box, in_dim).type(dtype).to(device) if hasattr(self, 'in_dim') else None

        null_input = {"boxes": boxes, "points": points, "masks": masks}
        if positive_embeddings is not None:
            null_input["positive_embeddings"] = positive_embeddings

        return null_input
