import torch

from models.encoders import VisualEncoder, TextEncoder
from models.fusion import ConcatFusion


visual_encoder = VisualEncoder(out_dim=256)
text_encoder = TextEncoder(
    n_instructions=3,
    out_dim=256,
)
fusion = ConcatFusion(embedding_dim=256)

images = torch.rand(4, 3, 64, 64)
instruction_ids = torch.tensor([0, 1, 2, 0])

visual_embedding = visual_encoder(images)
text_embedding = text_encoder(instruction_ids)

fused_embedding = fusion(
    visual_embedding,
    text_embedding,
)

print("Embedding visuel :", visual_embedding.shape)
print("Embedding texte :", text_embedding.shape)
print("Embedding fusionné :", fused_embedding.shape)