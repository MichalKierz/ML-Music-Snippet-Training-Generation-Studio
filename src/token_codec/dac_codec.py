from dataclasses import dataclass
from pathlib import Path

import dac
import torch
from audiotools import AudioSignal


@dataclass
class CodecSpec:
    sample_rate: int
    codebook_size: int
    n_codebooks: int
    pad_token_id: int
    bos_token_id: int
    eos_token_id: int
    vocab_size: int


class DACCodec:
    def __init__(self, model_type: str = "24khz", device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model_path = dac.utils.download(model_type=model_type)
        self.model = dac.DAC.load(self.model_path).to(self.device).eval()
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000))
        self.codebook_size = int(getattr(self.model, "codebook_size", 1024))
        self.n_codebooks = int(
            getattr(self.model, "n_codebooks", 0)
            or getattr(getattr(self.model, "quantizer", None), "n_codebooks", 0)
            or getattr(getattr(self.model, "quantizer", None), "num_codebooks", 0)
            or 0
        )

    def _ensure_codebook_count(self, codes: torch.Tensor):
        if self.n_codebooks <= 0:
            self.n_codebooks = int(codes.shape[0])

    def get_codec_spec(self) -> CodecSpec:
        base_vocab = int(self.codebook_size * self.n_codebooks)
        return CodecSpec(
            sample_rate=int(self.sample_rate),
            codebook_size=int(self.codebook_size),
            n_codebooks=int(self.n_codebooks),
            pad_token_id=base_vocab,
            bos_token_id=base_vocab + 1,
            eos_token_id=base_vocab + 2,
            vocab_size=base_vocab + 3,
        )

    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device).float()

        signal = AudioSignal(waveform, int(sample_rate))
        x = self.model.preprocess(signal.audio_data, signal.sample_rate)

        with torch.no_grad():
            _, codes, _, _, _ = self.model.encode(x)

        codes = codes.squeeze(0).detach().cpu().long()

        if codes.dim() != 2:
            raise RuntimeError(f"Unexpected DAC codes shape: {tuple(codes.shape)}")

        self._ensure_codebook_count(codes)
        return codes

    def flatten_codes(self, codes: torch.Tensor) -> torch.Tensor:
        if codes.dim() != 2:
            raise ValueError("codes must have shape [n_codebooks, time_steps]")

        n_codebooks, time_steps = codes.shape
        self._ensure_codebook_count(codes)

        offsets = torch.arange(n_codebooks, dtype=torch.long).unsqueeze(1) * int(self.codebook_size)
        global_codes = codes.long() + offsets
        flat = global_codes.transpose(0, 1).reshape(time_steps * n_codebooks)
        return flat.long()

    def unflatten_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.n_codebooks <= 0:
            raise RuntimeError("Codec n_codebooks is unknown. Encode at least one chunk first.")

        tokens = tokens.long().view(-1)
        usable = (tokens.numel() // self.n_codebooks) * self.n_codebooks
        tokens = tokens[:usable]

        if tokens.numel() == 0:
            raise RuntimeError("No usable audio tokens to decode.")

        time_steps = tokens.numel() // self.n_codebooks
        matrix = tokens.view(time_steps, self.n_codebooks).transpose(0, 1).contiguous()

        offsets = torch.arange(self.n_codebooks, dtype=torch.long).unsqueeze(1) * int(self.codebook_size)
        codes = matrix - offsets
        codes = torch.clamp(codes, min=0, max=int(self.codebook_size) - 1)
        return codes.long()

    def decode_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        codes = self.unflatten_tokens(tokens).unsqueeze(0).to(self.device)

        quantizer = getattr(self.model, "quantizer", None)
        if quantizer is None:
            raise RuntimeError("DAC quantizer was not found on the loaded model.")

        with torch.no_grad():
            from_codes_output = quantizer.from_codes(codes)

            if isinstance(from_codes_output, tuple):
                z = from_codes_output[0]
            else:
                z = from_codes_output

            audio = self.model.decode(z)

        audio = audio.squeeze(0).detach().cpu().float()

        if audio.dim() == 2 and audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        return audio

    def encode_file(self, path: str | Path) -> torch.Tensor:
        signal = AudioSignal(str(path))
        signal = signal.resample(self.sample_rate).to_mono()
        waveform = signal.audio_data.squeeze(0).detach().cpu()
        return self.encode_waveform(waveform, self.sample_rate)