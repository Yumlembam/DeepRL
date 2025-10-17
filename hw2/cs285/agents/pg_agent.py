from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # TODO: flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        obs_flat       = np.concatenate(obs, axis=0)           # list of (T_i, ob_dim)
        actions_flat   = np.concatenate(actions, axis=0)       # list of (T_i,) or (T_i, ac_dim)
        rewards_flat   = np.concatenate(rewards, axis=0)       # list of (T_i,)
        terminals_flat = np.concatenate(terminals, axis=0)     # list of (T_i,)
        qvals_flat     = np.concatenate(q_values, axis=0)      # list of (T_i,)

        if actions_flat.ndim == 2 and actions_flat.shape[1] == 1:
            actions_flat = actions_flat.squeeze(1)

        
        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs_flat, rewards_flat, qvals_flat, terminals_flat
        )
        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        actor_info: dict = self.actor.update(obs_flat, actions_flat, advantages)
        info: dict = {"actor":  actor_info["Actor Loss"]}

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs_flat, qvals_flat)
            info["critic"] = critic_info["Baseline Loss"]

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values = [self._discounted_return(r) for r in rewards]
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values = [self._discounted_reward_to_go(r) for r in rewards]

        return q_values

    # in PGAgent._estimate_advantage (drop-in replacement)
    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        # Convert once to torch (on correct device)
        obs_t       = ptu.from_numpy(obs)                   # only needed if you use critic
        rewards_t   = ptu.from_numpy(rewards).float()
        q_values_t  = ptu.from_numpy(q_values).float()
        terminals_t = ptu.from_numpy(terminals).float()     # 1.0 if terminal else 0.0

        if self.critic is None:
            adv_t = q_values_t.clone()
        else:
            with torch.no_grad():
                values_t = self.critic(obs_t).float()       # shape (N,)

            if self.gae_lambda is None:
                # simple baseline: A = Q - V
                adv_t = q_values_t - values_t
            else:
                # GAE(λ) with flat transitions & terminal flags
                gamma = torch.tensor(self.gamma, device=ptu.device, dtype=torch.float32)
                lam   = torch.tensor(self.gae_lambda, device=ptu.device, dtype=torch.float32)

                # Append dummy value for t = N (value_{N} = 0) to simplify indexing
                values_pad = torch.cat([values_t, torch.zeros(1, device=ptu.device)])
                adv_t = torch.zeros_like(q_values_t)
                gae = torch.tensor(0.0, device=ptu.device)

                # Compute deltas from rewards + V(s')
                # Here we don’t have next-state values directly, so we use the padded V
                # Terminals_t[i] == 1 → nonterminal = 0 → reset GAE at episode boundaries
                for i in range(q_values_t.numel() - 1, -1, -1):
                    nonterminal = 1.0 - terminals_t[i]
                    delta = rewards_t[i] + gamma * values_pad[i + 1] * nonterminal - values_pad[i]
                    gae = delta + gamma * lam * gae * nonterminal
                    adv_t[i] = gae

        if self.normalize_advantages:
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        return ptu.to_numpy(adv_t)  # keep your existing interface


    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        ret=0.0
        for r in reversed(rewards):
            ret=r+self.gamma*ret
        out=np.full(len(rewards),ret,dtype=np.float32)
        return out


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        ret=0.0
        rtg_list= []
        for r in reversed(rewards):
            ret=r+self.gamma*ret
            rtg_list.append(ret)
        rtg_list.reverse()
        out=np.array(rtg_list,dtype=np.float32)
        return out
