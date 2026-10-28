import torch
import torch.nn as nn

from models.encoders import VisualEncoder, TextEncoder
from models.fusion import ConcatFusion


visual_encoder = VisualEncoder(out_dim=256)
text_encoder = TextEncoder(
    n_instructions=3,
    out_dim=256,
)
fusion = ConcatFusion(embedding_dim=256)

classifier = nn.Linear(256, 3)

parameters = (
    list(visual_encoder.parameters())
    + list(text_encoder.parameters())
    + list(fusion.parameters())
    + list(classifier.parameters())
)

optimizer = torch.optim.Adam(
    parameters,
    lr=0.001,
)

images = torch.rand(8, 3, 64, 64)
instruction_ids = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

visual_embedding = visual_encoder(images)
text_embedding = text_encoder(instruction_ids)
fused_embedding = fusion(
    visual_embedding,
    text_embedding,
)

predictions = classifier(fused_embedding)

loss_function = nn.CrossEntropyLoss()
loss = loss_function(predictions, labels)

optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Forme des prédictions :", predictions.shape)
print("Valeur de la perte :", loss.item())
print("Étape d'entraînement réussie.")