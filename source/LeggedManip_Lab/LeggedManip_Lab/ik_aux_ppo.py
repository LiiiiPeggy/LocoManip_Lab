# Copyright (c) 2025-2026, Junjie Zhu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PPO with an auxiliary supervised term toward a local-IK teacher (stage 4.5e, variant C).

    L = L_PPO + lambda_ik * MSE( actor_mean(obs)[:, 2:], teacher(obs)[:, 2:] )

Only the six arm channels take part: channels 0-1 are the chassis velocity, which the teacher
says nothing about.

Why this re-implements `update()` instead of subclassing around it: rsl_rl 5.0.1's `PPO.update`
assembles the loss inside one method with no hook, and its `__init__` takes no `**kwargs`. The
two ways around that were to copy ~200 lines (which then silently drifts from the library) or
to reuse the "symmetry mirror loss", which is an MSE on the actor mean but would make the
training log and the config lie about what is being optimised. This is a deliberately
*reduced* update: no symmetry, no RND, single GPU. It asserts those are off rather than
pretending to support them, so there is nothing here that can drift out of sync -- what is
missing is missing by construction, and the assertion fails loudly if the config changes.

Chosen so that the auxiliary term is comparable in magnitude to the surrogate at the start
of training rather than dominating it; the coefficient is logged every update.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel


def build_mlp(obs_dim: int, device: str) -> nn.Module:
    """The teacher's architecture: the actor's own MLP, same widths and activation."""
    return nn.Sequential(
        nn.Linear(obs_dim, 512), nn.ELU(),
        nn.Linear(512, 256), nn.ELU(),
        nn.Linear(256, 128), nn.ELU(),
        nn.Linear(128, 8),
    ).to(device)


class PPOWithIKAux(PPO):
    """`:class:`rsl_rl.algorithms.PPO` plus a supervised term toward a frozen IK teacher."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage,
        ik_aux_coef: float = 0.5,
        ik_teacher_path: str = "",
        ik_arm_slice: tuple[int, int] = (2, 8),
        **kwargs,
    ):
        super().__init__(actor, critic, storage, **kwargs)
        if self.symmetry:
            raise RuntimeError("PPOWithIKAux does not implement the symmetry loss")
        if self.rnd:
            raise RuntimeError("PPOWithIKAux does not implement RND")
        if self.is_multi_gpu:
            raise RuntimeError("PPOWithIKAux does not implement multi-GPU reduction")
        self.ik_aux_coef = float(ik_aux_coef)
        self.ik_arm_slice = ik_arm_slice
        self.ik_teacher = None
        if ik_teacher_path and self.ik_aux_coef > 0.0:
            state = torch.load(ik_teacher_path, map_location=self.device, weights_only=True)
            # `teleop_ik_teacher.py` saves a plain nn.Sequential (`0.weight`); accept the
            # actor-checkpoint prefix (`mlp.0.weight`) too so either artefact works
            key0 = "mlp.0.weight" if "mlp.0.weight" in state else "0.weight"
            if key0 == "mlp.0.weight":
                state = {k[len("mlp."):]: v for k, v in state.items() if k.startswith("mlp.")}
            self.ik_teacher = build_mlp(state["0.weight"].shape[1], self.device)
            self.ik_teacher.load_state_dict(state)
            self.ik_teacher.eval()
            for p in self.ik_teacher.parameters():
                p.requires_grad_(False)
            print(f"[PPOWithIKAux] teacher loaded from {ik_teacher_path}, coef {self.ik_aux_coef}")
        elif self.ik_aux_coef > 0.0:
            raise RuntimeError("ik_aux_coef > 0 but no ik_teacher_path was given")

    def update(self) -> dict[str, float]:
        """PPO update plus the IK term, mirroring rsl_rl's loss composition."""
        mean_value = mean_surrogate = mean_entropy = 0.0
        mean_ik = 0.0
        mean_kl = 0.0
        num_updates = 0
        a0, a1 = self.ik_arm_slice

        for batch in self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            distribution_params = self.actor.output_distribution_params
            entropy = self.actor.output_entropy

            # adaptive learning rate on the measured KL, same rule as the parent class
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)
                    kl_mean = torch.mean(kl)
                    mean_kl += float(kl_mean.item())
                    if kl_mean > self.desired_kl * 2.0:
                        self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                    elif 0.0 < kl_mean < self.desired_kl / 2.0:
                        self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
            surrogate = -torch.squeeze(batch.advantages) * ratio
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.max((values - batch.returns).pow(2),
                                       (value_clipped - batch.returns).pow(2)).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            if self.ik_teacher is not None:
                # `batch.observations` is a TensorDict; the teacher is a plain MLP, so flatten
                # exactly the groups the actor reads (MLPModel.forward does the same concat)
                obs_flat = torch.cat([batch.observations[g] for g in self.actor.obs_groups], dim=-1)
                with torch.no_grad():
                    target = self.ik_teacher(obs_flat.detach())[:, a0:a1]
                mean_actions = self.actor(batch.observations.detach().clone())
                ik_loss = torch.mean((mean_actions[:, a0:a1] - target) ** 2)
                loss = loss + self.ik_aux_coef * ik_loss
                mean_ik += float(ik_loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value += float(value_loss.item())
            mean_surrogate += float(surrogate_loss.item())
            mean_entropy += float(entropy.mean().item())
            num_updates += 1

        self.storage.clear()
        out = {
            "value": mean_value / num_updates,
            "surrogate": mean_surrogate / num_updates,
            "entropy": mean_entropy / num_updates,
            # KL is not in the parent's dict; logging it is how the balance between the PPO
            # and auxiliary terms can be judged from the run, not from this docstring
            "kl": mean_kl / num_updates,
            "learning_rate": float(self.learning_rate),
        }
        if self.ik_teacher is not None:
            out["ik_aux"] = mean_ik / num_updates
        return out
