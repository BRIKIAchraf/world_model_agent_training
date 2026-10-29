import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):
    def __init__(
        self,
        action_dim=2,
        embed_dim=256,
        deter_dim=200,
        stoch_dim=32,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.embed_dim = embed_dim
        self.deter_dim = deter_dim
        self.stoch_dim = stoch_dim

        self.gru = nn.GRUCell(
            input_size=stoch_dim + action_dim,
            hidden_size=deter_dim,
        )

        self.prior_network = nn.Sequential(
            nn.Linear(deter_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),
        )

        self.posterior_network = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, 256),
            nn.ELU(),
            nn.Linear(256, 2 * stoch_dim),
        )

    def initial_state(self, batch_size, device=None):
        if device is None:
            device = next(self.parameters()).device

        deter = torch.zeros(
            batch_size,
            self.deter_dim,
            device=device,
        )

        stoch = torch.zeros(
            batch_size,
            self.stoch_dim,
            device=device,
        )

        return {
            "deter": deter,
            "stoch": stoch,
        }

    def _distribution(self, parameters):
        mean, log_std = torch.chunk(
            parameters,
            chunks=2,
            dim=-1,
        )

        std = F.softplus(log_std) + 0.1

        return torch.distributions.Normal(
            mean,
            std,
        )

    def features(self, state):
        return torch.cat(
            [
                state["deter"],
                state["stoch"],
            ],
            dim=-1,
        )

    def observe_step(
        self,
        state,
        action,
        observation_embedding,
    ):
        recurrent_input = torch.cat(
            [
                state["stoch"],
                action,
            ],
            dim=-1,
        )

        deter = self.gru(
            recurrent_input,
            state["deter"],
        )

        posterior_parameters = (
            self.posterior_network(
                torch.cat(
                    [
                        deter,
                        observation_embedding,
                    ],
                    dim=-1,
                )
            )
        )

        posterior_distribution = (
            self._distribution(
                posterior_parameters
            )
        )

        stoch = posterior_distribution.rsample()

        next_state = {
            "deter": deter,
            "stoch": stoch,
        }

        return next_state, {
            "posterior": posterior_distribution,
        }

    def imagine_step(self, state, action):
        recurrent_input = torch.cat(
            [
                state["stoch"],
                action,
            ],
            dim=-1,
        )

        deter = self.gru(
            recurrent_input,
            state["deter"],
        )

        prior_parameters = (
            self.prior_network(deter)
        )

        prior_distribution = (
            self._distribution(
                prior_parameters
            )
        )

        stoch = prior_distribution.rsample()

        next_state = {
            "deter": deter,
            "stoch": stoch,
        }

        return next_state, {
            "prior": prior_distribution,
        }