from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.torch_utils import FLOAT_MIN
from ray.rllib.utils.framework import try_import_torch
import sys

torch, nn = try_import_torch()


class ActionMaskedModel(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        # Move internal model to the specified device
        self.internal_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name
        )

    def forward(self, input_dict, state, seq_lens):
        # Move observations and action masks to device
        obs_data = input_dict["obs_flat"]
        action_mask =  input_dict["obs"]["action_mask"]

        # Pass the observation through the internal model and obtain logits
        logits, _ = self.internal_model({"obs": obs_data}, state, seq_lens)

        # Apply action mask to logits
        masked_logits = torch.where(action_mask == 1, logits, torch.tensor(-1e10))

        return masked_logits, state

    def value_function(self):
        # Ensure that the value function output is on the correct device
        return self.internal_model.value_function()
