from typing import Any

from omegaconf import DictConfig

import torch
import torch.nn as nn

from transformers import CLIPVisionModelWithProjection, CLIPTextModelWithProjection


def _freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


class CLIPVisionModel(nn.Module):
    def __init__(self, pretrained_name: str):
        super().__init__()

        original_model = CLIPVisionModelWithProjection.from_pretrained(pretrained_name)

        self.model = original_model
        self.vision_model = original_model.vision_model
        self.visual_projection = original_model.visual_projection
        self.config = original_model.config
        _freeze_module(self.model)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dense patch embeddings adapted from the last CLIP ViT block.

        The final block is truncated to the value pathway used by MaskCLIP:
        apply the block input LayerNorm, keep only the value projection, then
        pass it through the attention output projection. The last block's
        second LayerNorm and MLP are skipped for dense patch extraction.
        Patch tokens are finally projected to the CLIP latent space.
        """
        
        hidden_states = self.vision_model.embeddings(pixel_values=pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_layers = self.vision_model.encoder.layers
        for encoder_layer in encoder_layers[:-1]:
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        # For the last ViT block, keep only the value pathway from the final
        # attention layer for dense patch extraction.
        last_layer = encoder_layers[-1]

        hidden_states = last_layer.layer_norm1(hidden_states)
        hidden_states = last_layer.self_attn.v_proj(hidden_states)
        hidden_states = last_layer.self_attn.out_proj(hidden_states)

        # projection to image-text latent space
        projected_tokens = self.visual_projection(hidden_states)
        patch_embeddings = projected_tokens[:, 1:, :]
        return patch_embeddings


class CLIPVisionModel_v2(CLIPVisionModel):
    def _forward_to_final_block_input(
        self,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.vision_model.embeddings(pixel_values=pixel_values)
        hidden_states = self.vision_model.pre_layrnorm(hidden_states)

        encoder_layers = self.vision_model.encoder.layers
        for encoder_layer in encoder_layers[:-1]:
            layer_outputs = encoder_layer(
                hidden_states=hidden_states,
                attention_mask=None,
                causal_attention_mask=None,
                output_attentions=False,
            )
            if isinstance(layer_outputs, tuple):
                hidden_states = layer_outputs[0]
            else:
                hidden_states = layer_outputs

        return hidden_states

    def _forward_maskclip_dense_from_final_block_input(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        last_layer = self.vision_model.encoder.layers[-1]

        hidden_states = last_layer.layer_norm1(hidden_states)
        hidden_states = last_layer.self_attn.v_proj(hidden_states)
        hidden_states = last_layer.self_attn.out_proj(hidden_states)

        projected_tokens = self.visual_projection(hidden_states)
        return projected_tokens[:, 1:, :]

    def _forward_global_latent_from_final_block_input(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        last_layer = self.vision_model.encoder.layers[-1]
        layer_outputs = last_layer(
            hidden_states=hidden_states,
            attention_mask=None,
            causal_attention_mask=None,
            output_attentions=False,
        )
        if isinstance(layer_outputs, tuple):
            last_hidden_state = layer_outputs[0]
        else:
            last_hidden_state = layer_outputs

        pooled_output = self.vision_model.post_layernorm(last_hidden_state[:, 0, :])
        return self.visual_projection(pooled_output)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self._forward_to_final_block_input(pixel_values)
        patch_embeddings = self._forward_maskclip_dense_from_final_block_input(
            hidden_states
        )
        image_latent = self._forward_global_latent_from_final_block_input(hidden_states)
        return patch_embeddings, image_latent




class CLIPTextModel(nn.Module):
    def __init__(self, pretrained_name: str):

        super().__init__()

        original_model = CLIPTextModelWithProjection.from_pretrained(pretrained_name)
        self.text_model = original_model.text_model
        self.text_projection = original_model.text_projection
        _freeze_module(original_model)

    @staticmethod
    def _move_eos_to_front(
        hidden_states: torch.Tensor,
        eos_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        remaining_positions = torch.arange(seq_len - 1, device=hidden_states.device).unsqueeze(0).expand(batch_size, -1)
        source_positions = remaining_positions + (remaining_positions >= eos_token_indices.unsqueeze(1)).long()
        reordered_positions = torch.cat([eos_token_indices.unsqueeze(1), source_positions], dim=1)

        return hidden_states.gather(
            1,
            reordered_positions.unsqueeze(-1).expand(-1, -1, hidden_states.size(-1)),
        )

    def forward(self, **kwargs):
        output = self.text_model(**kwargs)

        last_hidden_state = output.last_hidden_state
        input_ids = kwargs["input_ids"]

        eos_token_indices = input_ids.argmax(dim=-1)
        batch_indices = torch.arange(last_hidden_state.size(0), device=last_hidden_state.device)
        eos_hidden_states = last_hidden_state[batch_indices, eos_token_indices]

        return self.text_projection(eos_hidden_states)    
