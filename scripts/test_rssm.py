import torch

from models.rssm import RSSM
from training.metrics import save_metrics


rssm = RSSM(
    action_dim=2,
    embed_dim=256,
    deter_dim=200,
    stoch_dim=32,
)

batch_size = 4

state = rssm.initial_state(
    batch_size=batch_size
)

action = torch.randn(
    batch_size,
    2,
)

observation_embedding = torch.randn(
    batch_size,
    256,
)

next_state, observe_info = rssm.observe_step(
    state,
    action,
    observation_embedding,
)

imagined_state, imagine_info = rssm.imagine_step(
    next_state,
    action,
)

features = rssm.features(
    imagined_state
)

save_metrics(
    "rssm_structure",
    {
        "experiment": "rssm_structure_test",
        "structure": {
            "batch_size": batch_size,
            "deter_dim": next_state["deter"].shape[-1],
            "stoch_dim": next_state["stoch"].shape[-1],
            "feature_dim": features.shape[-1],
        },
        "diagram": """digraph RSSM {
  rankdir=LR;
  action [label="Action\\n2"];
  stoch [label="Stoch\\n32"];
  gru [label="GRUCell"];
  deter [label="Deter\\n200"];
  features [label="Features\\n232"];
  action -> gru;
  stoch -> gru;
  gru -> deter;
  deter -> features;
  stoch -> features;
}""",
    },
)

print("Métriques RSSM sauvegardées dans outputs/metrics/")
