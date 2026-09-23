# SPDX-License-Identifier: Apache-2.0
import math

import torch
from mel import LogMel
from torch import nn
from torch.nn import functional as F


def load_model(checkpoint):
    from nemo.collections.asr.models import SortformerEncLabelModel

    model = SortformerEncLabelModel.restore_from(str(checkpoint), map_location="cpu", strict=True).eval()
    if not model.high_resolution or model.encoder.self_attention_model != "rope":
        raise ValueError("Expected the Nemotron-3 high-resolution RoPE diarizer")
    m = model.sortformer_modules
    m.chunk_len, m.chunk_right_context = 340, 40
    m.spkcache_len, m.fifo_len, m.spkcache_update_period = 264, 40, 300
    m._check_streaming_parameters()
    model.preprocessor.featurizer.dither = 0
    return model


def gather_frames(x, indices):
    return x.index_select(1, indices.flatten().clamp(0, x.shape[1] - 1))


def compress_cache(m, embeddings, predictions):
    # NeMo's AOSC selection, using functional scatter/where instead of dynamic boolean indexing.
    per_speaker = m.spkcache_len // m.n_spk - m.spkcache_sil_frames_per_spk
    scores = m._disable_low_scores(
        predictions, m._get_log_pred_scores(predictions), math.floor(per_speaker * m.min_pos_scores_rate)
    )
    latest = torch.arange(scores.shape[1], device=scores.device) >= m.spkcache_len
    scores = scores + latest[None, :, None] * m.scores_boost_latest
    for rate, scale in ((m.strong_boost_rate, 2), (m.weak_boost_rate, 1)):
        indices = scores.topk(math.floor(per_speaker * rate), dim=1, sorted=False).indices
        scores = scores.scatter(1, indices, scores.gather(1, indices) - scale * math.log(0.5))
    scores = F.pad(scores, (0, 0, 0, m.spkcache_sil_frames_per_spk), value=float("inf"))
    values, indices = scores.transpose(1, 2).flatten(1).topk(m.spkcache_len, sorted=False)
    indices = torch.where(values != float("-inf"), indices, m.max_index).sort(dim=1).values
    disabled = (indices == m.max_index) | (indices % scores.shape[1] >= predictions.shape[1])
    indices = torch.where(disabled, 0, indices % scores.shape[1])
    silence = m.learnable_sil_emb.to(embeddings.dtype)[None]
    return m._gather_spkcache_and_preds(embeddings, predictions, indices, disabled, silence)


class DiarizationStep(nn.Module):
    input_names = ["samples", "sample_bounds", "feature_length", "history", "history_preds", "history_length"]
    output_names = ["probabilities", "next_history", "next_history_preds"]

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.mel = LogMel(model.preprocessor.featurizer)
        for layer in model.encoder.layers:
            layer.attn.rope.extend_pe(684, "cpu", next(model.parameters()).dtype)

    def example_inputs(self):
        return (
            torch.zeros(1, LogMel.samples),
            torch.tensor([257, LogMel.samples]),
            torch.tensor([3040]),
            torch.zeros(1, 304, 512),
            torch.zeros(1, 264, 8),
            torch.tensor([0]),
        )

    def forward(self, samples, sample_bounds, feature_length, history, history_preds, history_length):
        features = self.mel(samples, sample_bounds, feature_length)
        model, m = self.model, self.model.sortformer_modules
        dtype = model.encoder.pre_encode.proj.weight.dtype
        chunk, chunk_length = model._call_pre_encode(features.to(dtype), feature_length)
        cache_length = history_length.clamp(max=264)
        fifo_length = history_length - cache_length
        positions = torch.arange(684, device=features.device)
        packed = torch.where(
            (positions < history_length)[None, :, None],
            gather_frames(history.to(dtype), positions),
            gather_frames(chunk, positions - history_length),
        )
        length = history_length + chunk_length
        valid = positions < length
        encoder = model.encoder
        x = encoder.embed_norm(packed)
        # NeMo's FlexAttention padding mask expressed as exportable SDPA.
        mask = torch.zeros(684, dtype=dtype, device=x.device).masked_fill(~valid, float("-inf"))
        mask = mask[None, None, None, :].expand(1, 1, 684, 684)
        for layer in encoder.layers:
            a = layer.attn
            q, k, v = a.w_qkv(layer.norm1(x)).reshape(1, 684, 3, 8, 64).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = a.rope(q, k)
            if dtype == torch.float32:
                attended = ((q @ k.transpose(-1, -2)) * (64**-0.5) + mask).softmax(-1) @ v
            else:
                attended = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            x = x + a.out_proj(attended.transpose(1, 2).reshape(1, 684, 512))
            x = x + layer.ffn(layer.norm2(x))
        x = m.encoder_proj(encoder.final_norm(x))
        # Padding must be zero before the upsampling convolution, including the final partial chunk.
        x = x * valid[None, :, None]
        probabilities = torch.sigmoid(m.forward_speaker_logits(m.upsample_hidden(x)))
        probabilities = probabilities * valid.repeat_interleave(8)[None, :, None]
        reduced = m.downsample_preds(probabilities, 8).float()
        output = gather_frames(probabilities, torch.arange(2720, device=x.device) + history_length * 8).float()

        # Every non-final offline chunk is full. The final state is deliberately unused.
        pop_length = fifo_length + 300
        candidate_length = cache_length + pop_length
        candidate_positions = torch.arange(604, device=x.device)
        candidate = gather_frames(packed, candidate_positions)
        candidate_preds = torch.where(
            (candidate_positions < cache_length)[None, :, None],
            gather_frames(history_preds, candidate_positions),
            gather_frames(reduced, candidate_positions),
        )
        candidate_preds = candidate_preds * (candidate_positions < candidate_length)[None, :, None]
        cache, cache_preds = compress_cache(m, candidate, candidate_preds)
        fifo = gather_frames(packed, torch.arange(40, device=x.device) + candidate_length)
        return output, torch.cat((cache, fifo), dim=1).float(), cache_preds.float()
