import torch
import torch.nn as nn

from samsone.data.constants import AUDIO_TO_SAMPLE_ASSIGNMENT_KEY
from samsone.models.base import Model
from samsone.models.projectors import Projector
from samsone.models.text.lm import LM


class AudioLM(Model):
    def __init__(
        self,
        audio_encoder: nn.Module,
        projector: Projector,
        text_model: LM,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.projector = projector
        self.text_model = text_model
        self.sep_token = nn.Parameter(torch.randn(projector.output_embedding_dim))

    def compact_sequence(self, sequence, mask):
        return sequence[torch.where(mask == 1)]

    def _combine_embeds_for_lm_input(
        self,
        projected_audio_embeddings: torch.Tensor,
        audio_attn_mask: torch.Tensor,
        metadata_batch: dict,
        pre_audio_text_embeddings: torch.Tensor,
        pre_audio_text_attention_mask: torch.Tensor,
        post_audio_tokens_batch: list[str],
        post_audio_text_embeddings: torch.Tensor,
        post_audio_text_attention_mask: torch.Tensor,
        response_mask_batch: torch.Tensor | None = None,
        max_tokens: int | None = None,
    ):
        batch_size = post_audio_text_embeddings.shape[0]
        is_training = response_mask_batch is not None

        concat_sequences = []
        attention_masks = []
        response_masks = [] if is_training else None
        response_tokens = [] if is_training else None
        max_seq_len = 0

        sample_indices = metadata_batch[AUDIO_TO_SAMPLE_ASSIGNMENT_KEY]

        # remove padding remaining from pre-audio text
        for i in range(batch_size):
            sample_audio_mask = sample_indices == i
            sample_audio_embeddings = projected_audio_embeddings[
                sample_audio_mask
            ]  # [num_audios_in_sample, seq_len, embed_dim]
            sample_audio_attn_mask = audio_attn_mask[
                sample_audio_mask
            ]  # [num_audios_in_sample, seq_len]

            sep_token_expanded = (
                self.sep_token.unsqueeze(0)
                .unsqueeze(1)
                .expand(
                    sample_audio_embeddings.size(0), 1, sample_audio_embeddings.size(2)
                )
            )  # [num_audios_in_sample, 1, embed_dim]

            sample_audio_embeddings = torch.cat(
                [sample_audio_embeddings, sep_token_expanded], dim=1
            )
            sample_audio_attn_mask = torch.cat(
                [
                    sample_audio_attn_mask,
                    torch.ones(
                        (sample_audio_attn_mask.shape[0], 1),
                        device=sample_audio_attn_mask.device,
                    ),
                ],
                dim=-1,
            )

            sample_audio_embeddings = torch.flatten(
                sample_audio_embeddings, start_dim=0, end_dim=1
            )  # [num_audios_in_sample * (seq_len + 1), embedding_dim]
            sample_audio_attn_mask = torch.flatten(
                sample_audio_attn_mask, start_dim=0, end_dim=1
            )  # [num_audios_in_sample * (seq_len + 1)]

            # select only relevant audio embeddings
            sample_audio_embeddings = sample_audio_embeddings[
                sample_audio_attn_mask.bool()
            ]

            sample_audio_embeddings = torch.cat(
                [self.sep_token.unsqueeze(0), sample_audio_embeddings], dim=0
            )  # add sep_token before audio

            pre_seq = self.compact_sequence(
                pre_audio_text_embeddings[i], pre_audio_text_attention_mask[i]
            )
            seq = torch.cat([pre_seq, sample_audio_embeddings], dim=0)

            # remove padding from post-audio text
            post_seq = self.compact_sequence(
                post_audio_text_embeddings[i], post_audio_text_attention_mask[i]
            )
            post_attn_mask = self.compact_sequence(
                post_audio_text_attention_mask[i], post_audio_text_attention_mask[i]
            )

            # handle response masks only for training
            if is_training:
                # find response tokens before removing padding
                idx = torch.where(response_mask_batch[i] == 1)[0]
                start, end = idx[0].item(), idx[-1].item() + 1
                response_tokens.append(
                    post_audio_tokens_batch["input_ids"][i, start:end]
                )

                # remove padding from response mask to match cleaned post-audio text
                response_mask_cleaned = self.compact_sequence(
                    response_mask_batch[i], post_audio_text_attention_mask[i]
                )

                # prepend zeros to response masks that match the length of pre_audio_embeddings + audio embeddings
                response_mask = torch.cat(
                    [
                        torch.zeros(
                            seq.shape[0],
                            dtype=response_mask_batch.dtype,
                            device=projected_audio_embeddings.device,
                        ),
                        response_mask_cleaned,
                    ],
                    dim=0,
                )
                response_masks.append(response_mask)

            # create full sequence attention masks (to be padded due to variable length)
            attention_masks.append(
                torch.cat(
                    [
                        torch.ones(
                            seq.shape[0],
                            dtype=post_audio_text_attention_mask.dtype,
                            device=projected_audio_embeddings.device,
                        ),
                        post_attn_mask,
                    ],
                    dim=0,
                )
            )

            concat_seq = torch.cat([seq, post_seq])
            concat_sequences.append(concat_seq)

            if concat_seq.shape[0] > max_seq_len:
                max_seq_len = concat_seq.shape[0]

        emb_size = pre_audio_text_embeddings.shape[-1]

        combined_embeds = torch.zeros(
            batch_size,
            max_seq_len,
            emb_size,
            dtype=projected_audio_embeddings.dtype,
            device=projected_audio_embeddings.device,
        )
        combined_attn_mask = torch.zeros(
            batch_size,
            max_seq_len,
            dtype=torch.long,
            device=projected_audio_embeddings.device,
        )

        combined_labels = None
        if is_training:
            combined_labels = (
                torch.ones(
                    batch_size,
                    max_seq_len,
                    dtype=torch.long,
                    device=projected_audio_embeddings.device,
                )
                * -100
            )

        # pad the end again (to account for removed padding from pre-audio text)
        for i in range(batch_size):
            if is_training:
                seq_len = concat_sequences[i].shape[0]
                combined_embeds[i, :seq_len] = concat_sequences[i]
                combined_attn_mask[i, :seq_len] = attention_masks[i]

                idx = torch.where(response_masks[i] == 1)[0]
                start, end = idx[0].item(), idx[-1].item() + 1
                combined_labels[i, start:end] = response_tokens[i]
            else:
                seq_len = concat_sequences[i].shape[0]
                start_token = combined_embeds.shape[1] - seq_len
                combined_embeds[i, start_token:] = concat_sequences[i]
                combined_attn_mask[i, start_token:] = attention_masks[i]

        if max_tokens:
            combined_embeds = combined_embeds[:, :max_tokens, :]
            combined_attn_mask = combined_attn_mask[:, :max_tokens]
            if is_training:
                combined_labels = combined_labels[:, :max_tokens]

        if is_training:
            return combined_embeds, combined_attn_mask, combined_labels
        else:
            return combined_embeds, combined_attn_mask

    def forward(
        self,
        audio_features_batch: dict,
        metadata_batch: dict,
        pre_audio_tokens_batch: list[str],
        post_audio_tokens_batch: list[str],
        response_mask_batch: torch.Tensor,
        max_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        audio_embeddings, audio_attn_mask = self.audio_encoder(audio_features_batch)
        projected_audio_embeddings, audio_attn_mask = self.projector(
            audio_embeddings, audio_attn_mask
        )  # [all_audios_in_batch, audio_encoder_seq_len, embedding_dim]

        pre_audio_text_embeddings = self.text_model.get_embeddings_from_tokens(
            pre_audio_tokens_batch["input_ids"]
        )
        pre_audio_text_attention_mask = pre_audio_tokens_batch["attention_mask"]

        post_audio_text_embeddings = self.text_model.get_embeddings_from_tokens(
            post_audio_tokens_batch["input_ids"]
        )
        post_audio_text_attention_mask = post_audio_tokens_batch["attention_mask"]

        combined_embeds, combined_attn_mask, combined_labels = (
            self._combine_embeds_for_lm_input(
                projected_audio_embeddings,
                audio_attn_mask,
                metadata_batch,
                pre_audio_text_embeddings,
                pre_audio_text_attention_mask,
                post_audio_tokens_batch,
                post_audio_text_embeddings,
                post_audio_text_attention_mask,
                response_mask_batch,
                max_tokens,
            )
        )

        model_output = self.text_model.process_embeddings(
            embeddings=combined_embeds,
            attention_mask=combined_attn_mask,
            labels=combined_labels,
        )

        return model_output, combined_attn_mask

    def generate(
        self,
        audio_features_batch: dict,
        metadata_batch: dict,
        pre_audio_tokens_batch: list[str],
        post_audio_tokens_batch: list[str],
        max_new_tokens,
        **kwargs,
    ) -> list[str]:
        audio_embeddings, audio_attn_mask = self.audio_encoder(audio_features_batch)
        projected_audio_embeddings, audio_attn_mask = self.projector(
            audio_embeddings, audio_attn_mask
        )  # [all_audios_in_batch, audio_encoder_seq_len, embedding_dim]

        pre_audio_text_embeddings = self.text_model.get_embeddings_from_tokens(
            pre_audio_tokens_batch["input_ids"]
        )
        pre_audio_text_attention_mask = pre_audio_tokens_batch["attention_mask"]

        post_audio_text_embeddings = self.text_model.get_embeddings_from_tokens(
            post_audio_tokens_batch["input_ids"]
        )
        post_audio_text_attention_mask = post_audio_tokens_batch["attention_mask"]

        combined_embeds, combined_attn_mask = self._combine_embeds_for_lm_input(
            projected_audio_embeddings=projected_audio_embeddings,
            audio_attn_mask=audio_attn_mask,
            metadata_batch=metadata_batch,
            pre_audio_text_embeddings=pre_audio_text_embeddings,
            pre_audio_text_attention_mask=pre_audio_text_attention_mask,
            post_audio_text_embeddings=post_audio_text_embeddings,
            post_audio_text_attention_mask=post_audio_text_attention_mask,
            post_audio_tokens_batch=post_audio_tokens_batch,
            response_mask_batch=None,  # No response mask for generation
            max_tokens=None,
        )

        generated_text = self.text_model.generate(
            input_embeddings=combined_embeds,
            attention_mask=combined_attn_mask,
            max_new_tokens=max_new_tokens,
            **kwargs,
        )

        return generated_text


class AudioLMForExport(Model):
    def __init__(
        self,
        audio_encoder: nn.Module,
        projector: Projector,
        text_model: LM,
    ):
        super().__init__()
        self.audio_encoder = audio_encoder
        self.projector = projector
        self.text_model = text_model
        self.sep_token = nn.Parameter(torch.randn(projector.output_embedding_dim))

    def _combine_embeds_for_lm_input(
        self,
        projected_audio_embeddings: torch.Tensor,
        audio_attn_mask: torch.Tensor,
        post_audio_text_embeddings: torch.Tensor,
    ):
        sample_audio_embeddings = (
            projected_audio_embeddings  # [num_audios_in_sample, seq_len, embed_dim]
        )

        sep_token_expanded = (
            self.sep_token.unsqueeze(0)
            .unsqueeze(1)
            .expand(sample_audio_embeddings.size(0), 1, sample_audio_embeddings.size(2))
        )  # [num_audios_in_sample, 1, embed_dim]

        sample_audio_embeddings = torch.cat(
            [sample_audio_embeddings, sep_token_expanded], dim=1
        )

        audio_attn_mask = torch.cat(
            [
                audio_attn_mask,
                torch.ones(
                    (audio_attn_mask.shape[0], 1),
                    device=audio_attn_mask.device,
                ),
            ],
            dim=-1,
        )

        sample_audio_embeddings = torch.flatten(
            sample_audio_embeddings, start_dim=0, end_dim=1
        )  # [num_audios_in_sample * (seq_len + 1), embedding_dim]

        audio_attn_mask = torch.flatten(
            audio_attn_mask, start_dim=0, end_dim=1
        )  # [num_audios_in_sample * (seq_len + 1)]

        # select only relevant audio embeddings
        # sample_audio_embeddings = sample_audio_embeddings[audio_attn_mask.bool()] # TODO audio masking doesn't work now
        sample_audio_embeddings = torch.cat(
            [self.sep_token.unsqueeze(0), sample_audio_embeddings],
            dim=0,  # add sep_token before audio
        ).unsqueeze(0)

        combined_embeds = torch.cat(
            [sample_audio_embeddings, post_audio_text_embeddings], dim=1
        )

        return combined_embeds

    def forward(
        self,
        audio_features_batch: dict,
        post_audio_tokens_batch: list[str],
        **kwargs,
    ) -> list[str]:
        audio_embeddings, audio_attn_mask = self.audio_encoder(audio_features_batch)
        projected_audio_embeddings, audio_attn_mask = self.projector(
            audio_embeddings, audio_attn_mask
        )  # [all_audios_in_batch, audio_encoder_seq_len, embedding_dim]

        post_audio_text_embeddings = self.text_model.get_embeddings_from_tokens(
            post_audio_tokens_batch["input_ids"]
        )

        combined_embeds = self._combine_embeds_for_lm_input(
            projected_audio_embeddings=projected_audio_embeddings,
            audio_attn_mask=audio_attn_mask,
            post_audio_text_embeddings=post_audio_text_embeddings,
        )

        return combined_embeds
