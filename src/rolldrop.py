"""C3 policy: dropout on during PPO rollouts, off during updates.

This file changes the network, not the sensors. C3 still uses a clean env.
C0-C2 never import this; they use the default MlpPolicy.

Who calls it:

- src/train.py make_model, when rolldrop.dropout_p > 0
- scripts/sweep_rolldrop.py, same class at each p
- src/evaluate.py, import RollDropPolicy before PPO.load for a C3 checkpoint

How it hooks into PPO:

    collect_rollouts -> policy.forward()          dropout ON
    PPO.train        -> policy.evaluate_actions() dropout OFF (not overridden)
    predict / eval   -> policy.predict()          dropout OFF (not overridden)

Dropout sits after the hidden layers, on the actor latent only.
"""

from stable_baselines3.common.policies import ActorCriticPolicy
from torch.nn import functional as F

""" Old VS New Code, and why PPO cares which one you give it.

PPO does not read a list of steps from this file. It only talks to a
policy object: policy.forward() while playing, policy.evaluate_actions()
while learning. If C3 is not that object, PPO never sees it.

Old: RollDrop.drop_transitions(batch) deleted whole practice steps.
     PPO has no call for that, so train.py never passed it in.

New: RollDropPolicy IS the policy. Dropout runs inside forward(), on
     actor hidden units, not on the dataset.
"""


def apply_roll_drop(latent, dropout_p):
    """Randomly zero hidden units. Identity when dropout_p is 0. This will give the 

    Input: latent tensor from the actor MLP, dropout_p in [0, 1).
    - Latent is a tensor (N-dimensional array) of shape (batch_size, hidden_units)
    - dropout_p is a float in [0, 1) that controls the dropout rate

    Output: tensor of the same shape.
    """
    if dropout_p == 0.0:
        return latent
    # This is the actual random drop: each unit is zeroed with probability p. Use F from torch.nn for dropout.
    return F.dropout(latent, p=dropout_p, training=True)


class RollDropPolicy(ActorCriticPolicy):
    """Same MLP as C0-C2, plus rollout-only dropout on the actor."""

    def __init__(self, *args, dropout_p=0.1, **kwargs):
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
        self.dropout_p = float(dropout_p)
        super().__init__(*args, **kwargs)

    def forward(self, obs, deterministic=False):
        """Used only while PPO collects rollouts.

        1. Turn the observation into features.
        2. Run the shared or split MLP.
        3. Drop actor units (this is Roll-Drop).
        4. Sample an action and a value.
        """
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        # Actor only. Critic latent_vf is left intact.
        latent_pi = apply_roll_drop(latent_pi, self.dropout_p)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob
