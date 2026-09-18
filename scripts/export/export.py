from datetime import datetime
import json
import os
import subprocess

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import (
    EdgeCompileConfig,
    to_edge_transform_and_lower,
)
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.export import Dim
import torch.nn as nn
import torch.nn.functional as F
import yaml

from samsone import SamsoneCLI

# WhisperAudioEncoder copied from executorch/extension/audio/mel_spectrogram.py


def get_current_time():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")


OmegaConf.register_new_resolver("now", get_current_time)


def initialize_model_from_config(config_path):
    cli = SamsoneCLI(
        run=False,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={"parser_mode": "omegaconf"},
        auto_configure_optimizers=False,
        args=["--config", config_path],
    )

    audio_lm_model = cli.model.audio_lm
    if dtype == torch.float16:
        audio_lm_model.half().eval()
    return audio_lm_model


def extract_export_llm_config(
    cfg: DictConfig, output_path: str = "export_llm_config.yaml"
):
    """Extract export_llm configuration and save to separate YAML file"""
    export_llm_config = OmegaConf.select(cfg, "export_llm_config")
    if export_llm_config:
        # Convert to regular dict and save as YAML
        config_dict = OmegaConf.to_container(export_llm_config, resolve=True)
        with open(output_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Export LLM config saved to {output_path}")
    else:
        print("No export_llm_config found in main config")


class EmbeddingModel(torch.nn.Module):
    def __init__(self, audio_lm):
        super().__init__()
        self.audio_lm = audio_lm

    def forward(self, tokens):
        return self.audio_lm.text_model.get_embeddings_from_tokens(tokens)


class ArgMaxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits):
        return torch.argmax(logits, dim=1)[0]


def convert_tm_weights(original_model_folder_path, converted_model_path):
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "executorch.examples.models.smollm2.convert_weights",
        original_model_folder_path,
        converted_model_path,
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Success!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"Stderr: {e.stderr}")


def export_lm():
    command = [
        "uv",
        "run",
        "python",
        "-m",
        "executorch.extension.llm.export.export_llm",
        "--config",
        "scripts/export/export_llm_config.yaml",
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("Success!")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")
        print(f"Stderr: {e.stderr}")


class CropOrPad(nn.Module):
    """
    Input shape: [time]
    Output shape: [segment_length]
    """

    def __init__(self, segment_length: int):
        super().__init__()
        self.segment_length = segment_length

    def forward(self, waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        diff = waveform.shape[-1] - self.segment_length

        if diff > 0:
            # Crop
            output = waveform[: self.segment_length]
        elif diff < 0:
            # Pad
            pad_len = -diff
            pad = (0, pad_len)
            output = F.pad(waveform, pad, value=0.0)
        else:
            # Length is already correct
            output = waveform

        return output


class WhisperAudioProcessor(nn.Module):
    r"""
    Computes Mel spectrograms from mono audio input.
    Same as HuggingFace WhisperFeatureExtractor, but implemented in PyTorch

    Args:
        feature_size (`int`, defaults to 80):
            The feature dimension of the extracted features.
        sampling_rate (`int`, defaults to 16000):
            The sampling rate at which the audio files should be digitalized expressed in hertz (Hz).
        hop_length (`int`, defaults to 160):
            Length of the overlaping windows for the STFT used to obtain the Mel Frequency coefficients.
        chunk_length (`int`, defaults to 30):
            The maximum number of chuncks of `sampling_rate` samples used to trim and pad longer or shorter audio
            sequences.
        n_fft (`int`, defaults to 400):
            Size of the Fourier transform.
        padding_value (`float`, *optional*, defaults to 0.0):
            Padding value used to pad the audio. Should correspond to silences.
    """

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        hop_length: int = 160,
        chunk_length: int = 30,
        n_fft: int = 400,
        padding_value: float = 0.0,
        stack_output: bool = False,
    ) -> None:
        super().__init__()
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.padding_value = padding_value

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.chunk_length = chunk_length
        self.n_samples = chunk_length * sampling_rate
        self.crop_or_pad = CropOrPad(self.n_samples)
        self.nb_max_frames = self.n_samples // hop_length
        self.sampling_rate = sampling_rate
        self.mel_filters = self.get_mel_filters(
            sampling_rate, n_fft, n_mels=feature_size
        )
        self.stack_output = stack_output

    def get_mel_filters(
        self,
        sr: int,
        n_fft: int,
        n_mels: int = 128,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # Initialize the weights
        n_mels = int(n_mels)
        weights = torch.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = torch.fft.rfftfreq(n=n_fft, d=1.0 / sr, dtype=dtype)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = torch.linspace(min_mel, max_mel, n_mels + 2, dtype=dtype)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = (
            torch.log(torch.tensor(6.4, dtype=dtype)) / 27.0
        )  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = torch.diff(mel_f)
        ramps = torch.subtract(mel_f.unsqueeze(1), fftfreqs.unsqueeze(0))

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = torch.maximum(
                torch.tensor(0.0, dtype=dtype), torch.minimum(lower, upper)
            )

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])  # pyre-ignore[58]
        weights *= enorm[:, None]  # pyre-ignore[16]

        return weights

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        r"""
        Args:
            waveform (`torch.Tensor`): Mono waveform input, tensor of (dynamic) shape [num_samples],

        Returns:
            torch.Tensor: Output of shape [1, feature_size, nb_max_frames * n_chunks]
            n_chunks is the number of chunks of `sampling_rate` samples in the input waveform.
            [1, 80, 3000] with default options and 1 chunk
        """
        waveform = self.crop_or_pad(waveform)

        # Ideally we should do:
        # window = torch.hann_window(self.n_fft)
        # but this is not currently supported when lowering.
        # torch.hann_window has slightly better numerics (worst discrepancy is <1e-5 instead of 1e-4)
        window = 0.5 * (
            1
            - torch.cos(
                2
                * torch.pi
                * torch.linspace(0, self.n_fft - 1, self.n_fft, dtype=torch.float32)
                / self.n_fft
            )
        )
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            center=True,
            return_complex=True,
        )
        magnitudes = torch.abs(stft)[..., :-1] ** 2  # pyre-ignore[58]

        mel_spec = self.mel_filters @ magnitudes

        log_spec = torch.log10(torch.clamp(mel_spec, min=1e-10))
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        if self.stack_output:
            log_spec = log_spec.reshape(self.feature_size, -1, self.nb_max_frames)
            log_spec = log_spec.transpose(0, 1)
            return log_spec.to(dtype)
        else:
            return log_spec.unsqueeze(0).to(dtype)


def export_processor(cfg):
    cfg = cfg.audio_processor
    model = WhisperAudioProcessor(
        feature_size=cfg.feature_size,
        sampling_rate=cfg.sampling_rate,
        hop_length=cfg.hop_length,
        chunk_length=cfg.chunk_length,
        n_fft=cfg.n_fft,
        stack_output=cfg.stack_output,
    )
    # if dtype==torch.float16:
    #     model.half().eval()

    audio_tensor = torch.randn(93680, dtype=torch.float32)
    shapes_collection = torch.export.ShapesCollection()
    max_n_chunks = 480000
    shapes_collection[audio_tensor] = {0: Dim.DYNAMIC(max=max_n_chunks)}
    with (
        torch.no_grad(),
        torch.fx.experimental._config.patch(backed_size_oblivious=True),
    ):
        ep = torch.export.export(
            model, (audio_tensor,), dynamic_shapes=shapes_collection, strict=True
        )

    return ep


def export_models(model, cfg):
    lm_config_json_path = cfg.export_llm_config.base.params
    audio_features_shape = cfg.audio_export.audio_features_shape
    post_audio_tokens_shape = cfg.audio_export.post_audio_tokens_shape
    with open(lm_config_json_path, "r") as f:
        lm_params = json.load(f)
        vocab_size = lm_params["vocab_size"]

    audio_features_batch = {
        "input_features": torch.randn(*audio_features_shape, dtype=dtype)
    }

    post_audio_tokens_batch = {
        "input_ids": torch.ones(list(post_audio_tokens_shape), dtype=torch.int),
    }

    example_inputs = (
        audio_features_batch,
        post_audio_tokens_batch,
    )

    dynamic_shapes = (
        {"input_features": {0: torch.export.Dim("num_audios_dim", min=1, max=2)}},
        {"input_ids": {1: torch.export.Dim("token_dim", max=400)}},
    )

    print("Exporting Audio Model...")
    exported_audio_lm = torch.export.export(
        model, example_inputs, dynamic_shapes=dynamic_shapes, strict=True
    )

    print("Exporting Embedding Model...")
    emb_model = EmbeddingModel(model)
    example_emb_model_inputs = (
        torch.ones(list(post_audio_tokens_shape), dtype=torch.int),
    )
    dynamic_emb_model_shapes = ({1: torch.export.Dim("token_dim", max=400)},)
    exported_emb_model = torch.export.export(
        emb_model,
        example_emb_model_inputs,
        dynamic_shapes=dynamic_emb_model_shapes,
        strict=True,
    )

    print("Exporting ArgMax Model...")
    argmax_model = ArgMaxModel()
    example_argmax_model_inputs = (torch.randn((1, vocab_size), dtype=dtype),)
    exported_argmax_model = torch.export.export(
        argmax_model,
        example_argmax_model_inputs,
        strict=True,
    )

    print("Exporting Audio Processor...")
    exported_audio_processor = export_processor(cfg)

    return (
        exported_audio_lm,
        exported_emb_model,
        exported_argmax_model,
        exported_audio_processor,
    )


def to_edge(
    cfg,
    exported_audio_lm,
    exported_emb_model,
    exported_argmax_model,
    exported_audio_processor,
):
    output_file = cfg.audio_export.output_file

    output_dir = os.path.dirname(output_file)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print("To Edge and Lower...")
    program = to_edge_transform_and_lower(
        {
            "forward": exported_audio_lm,
            "embedding": exported_emb_model,
            "argmax": exported_argmax_model,
            "processor": exported_audio_processor,
        },
        partitioner=[XnnpackPartitioner()],
        compile_config=EdgeCompileConfig(
            _check_ir_validity=False,
        ),
    ).to_executorch()

    # Save exported model
    with open(output_file, "wb") as f:
        f.write(program.buffer)

    print(f"Audio model exported to {output_file}")


dtype = None


@hydra.main(version_base=None, config_path=".", config_name="config100M")
def main(cfg: DictConfig) -> None:
    global dtype
    dtype = torch.float32 if cfg.dtype == "fp32" else torch.float16

    export_llm_config_path = cfg.export_llm_config_path
    extract_export_llm_config(cfg, export_llm_config_path)
    config_path = cfg.audio_export.config_path

    model = initialize_model_from_config(config_path).eval()
    # Save text model
    text_model_dir = cfg.text_model_dir
    os.makedirs(text_model_dir, exist_ok=True)
    tm = model.text_model.model
    tm.save_pretrained(text_model_dir)
    del model.text_model.model

    exported_models = export_models(model, cfg)
    to_edge(cfg, *exported_models)

    print("Converting text model weights...")
    convert_tm_weights(cfg.text_model_dir, f"{text_model_dir}/converted.pt")

    print("Exporting language model...")
    export_lm()


if __name__ == "__main__":
    main()
