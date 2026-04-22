from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset, random_split

from src.core.task_cancelled import TaskCancelledError
from src.data.token_sequence_dataset import TokenSequenceDataset, collate_token_batch
from src.models.metadata_token_transformer import MetadataTokenTransformer
from src.training.token_batch_sampler import (
    BucketedTokenBatchSampler,
    estimate_bucket_size_multiplier,
    estimate_max_examples_per_batch,
    estimate_token_budget,
)
from src.training.token_trainer import TokenTrainer
from src.training.token_training_service_utils import (
    build_model_file_name,
    build_warmup_cosine_scheduler,
    count_model_parameters,
    load_codec_info,
    release_torch_memory,
    resolve_sequence_lengths,
    resolve_training_setup,
    save_training_payload,
)


def _resolve_subset_lengths(data_source, max_token_length: int) -> list[int]:
    max_payload_length = max(1, int(max_token_length) - 1)

    if isinstance(data_source, Subset):
        base_dataset = data_source.dataset
        indices = [int(index) for index in data_source.indices]
        token_lengths = base_dataset.df.iloc[indices]["token_length"].astype(int).tolist()
    else:
        token_lengths = data_source.df["token_length"].astype(int).tolist()

    return [min(int(length), max_payload_length) + 1 for length in token_lengths]


class TokenTrainingService:
    def __init__(self, token_model_defaults: dict, training_defaults: dict):
        self.token_model_defaults = dict(token_model_defaults)
        self.training_defaults = dict(training_defaults)

    def run(
        self,
        token_manifest_path: str | Path,
        codec_info_path: str | Path,
        metadata_vocab_path: str | Path,
        model_output_folder: str | Path,
        batch_size: int,
        epochs: int,
        learning_rate: float,
        validation_split: float,
        weight_decay: float,
        mixed_precision: bool,
        pin_memory: bool,
        use_genre_metadata: bool,
        use_year_metadata: bool,
        use_mood_metadata: bool = False,
        use_key_metadata: bool = False,
        use_bpm_metadata: bool = False,
        training_gpu_ids=None,
        d_model: int = 512,
        n_heads: int = 8,
        n_kv_heads: int | None = None,
        n_layers: int = 8,
        ff_mult: int = 4,
        model_dropout: float = 0.1,
        max_token_length: int | None = None,
        metadata_prefix_tokens: int = 8,
        reference_track_duration_sec: float = 240.0,
        rope_base: float = 10000.0,
        processed_training_data_folder: str | Path | None = None,
        progress_callback=None,
        is_cancelled=None,
    ) -> dict:
        dataset = None
        train_loader = None
        val_loader = None
        model = None
        optimizer = None
        scheduler_package = None

        def ensure_not_cancelled():
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token training cancelled.")

        try:
            resolved_n_heads = int(n_heads)
            resolved_n_kv_heads = int(n_kv_heads if n_kv_heads is not None else n_heads)

            if int(d_model) % resolved_n_heads != 0:
                raise ValueError("Model Dimension must be divisible by Attention Heads.")

            if resolved_n_heads % resolved_n_kv_heads != 0:
                raise ValueError("Attention Heads must be divisible by KV Heads.")

            token_manifest_path = Path(token_manifest_path)
            codec_info_path = Path(codec_info_path)
            metadata_vocab_path = Path(metadata_vocab_path)
            model_output_folder = Path(model_output_folder)
            model_output_folder.mkdir(parents=True, exist_ok=True)

            codec_info = load_codec_info(codec_info_path)
            length_info, _ = resolve_sequence_lengths(token_manifest_path)
            derived_max_token_length = int(length_info["derived_max_token_length"])

            ensure_not_cancelled()

            dataset = TokenSequenceDataset(
                token_manifest_path=token_manifest_path,
                codec_info=codec_info,
                metadata_encoder=None,
                build_metadata_encoder=True,
                metadata_vocab_path=metadata_vocab_path,
                title_max_length=64,
                max_token_length=derived_max_token_length,
                use_genre_metadata=bool(use_genre_metadata),
                use_year_metadata=bool(use_year_metadata),
                use_mood_metadata=bool(use_mood_metadata),
                use_key_metadata=bool(use_key_metadata),
                use_bpm_metadata=bool(use_bpm_metadata),
            )

            total_len = len(dataset)
            if total_len == 0:
                raise ValueError("Token dataset is empty.")

            val_len = int(round(total_len * float(validation_split)))
            val_len = min(max(val_len, 0), max(0, total_len - 1))
            train_len = total_len - val_len

            if val_len > 0:
                generator = torch.Generator().manual_seed(42)
                train_dataset, val_dataset = random_split(
                    dataset,
                    [train_len, val_len],
                    generator=generator,
                )
            else:
                train_dataset = dataset
                val_dataset = None

            primary_device, resolved_gpu_ids = resolve_training_setup(training_gpu_ids)
            effective_pin_memory = bool(pin_memory) and primary_device.type == "cuda"

            train_lengths = _resolve_subset_lengths(train_dataset, derived_max_token_length)
            reference_sequence_length = max(1, int(length_info["median_token_length"]) + 1)
            token_budget_per_batch = int(
                self.training_defaults.get(
                    "token_budget_per_batch",
                    estimate_token_budget(
                        reference_batch_size=int(batch_size),
                        reference_sequence_length=reference_sequence_length,
                        min_budget=reference_sequence_length,
                    ),
                )
            )
            max_examples_per_batch = int(
                self.training_defaults.get(
                    "max_examples_per_batch",
                    estimate_max_examples_per_batch(
                        reference_batch_size=int(batch_size),
                        expansion_factor=int(self.training_defaults.get("max_examples_per_token_batch_factor", 4)),
                    ),
                )
            )
            bucket_size_multiplier = int(
                self.training_defaults.get(
                    "bucket_size_multiplier",
                    estimate_bucket_size_multiplier(total_len),
                )
            )

            train_batch_sampler = BucketedTokenBatchSampler(
                lengths=train_lengths,
                max_tokens_per_batch=token_budget_per_batch,
                max_examples_per_batch=max_examples_per_batch,
                shuffle=True,
                bucket_size_multiplier=bucket_size_multiplier,
                seed=42,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_sampler=train_batch_sampler,
                num_workers=int(self.training_defaults.get("num_workers", 0)),
                pin_memory=effective_pin_memory,
                collate_fn=lambda batch: collate_token_batch(batch, dataset.pad_token_id),
            )

            if val_dataset is not None:
                val_lengths = _resolve_subset_lengths(val_dataset, derived_max_token_length)
                val_batch_sampler = BucketedTokenBatchSampler(
                    lengths=val_lengths,
                    max_tokens_per_batch=token_budget_per_batch,
                    max_examples_per_batch=max_examples_per_batch,
                    shuffle=False,
                    bucket_size_multiplier=bucket_size_multiplier,
                    seed=42,
                )
                val_loader = DataLoader(
                    val_dataset,
                    batch_sampler=val_batch_sampler,
                    num_workers=int(self.training_defaults.get("num_workers", 0)),
                    pin_memory=effective_pin_memory,
                    collate_fn=lambda batch: collate_token_batch(batch, dataset.pad_token_id),
                )

            metadata_encoder = dataset.metadata_encoder

            model_config = {
                "architecture_name": "token_decoder_rope_cache",
                "architecture_version": 1,
                "token_vocab_size": int(dataset.vocab_size),
                "artist_vocab_size": int(metadata_encoder.artist_vocab_size),
                "genre_vocab_size": int(metadata_encoder.genre_vocab_size),
                "mood_vocab_size": int(metadata_encoder.mood_vocab_size),
                "key_vocab_size": int(metadata_encoder.key_vocab_size),
                "title_vocab_size": int(metadata_encoder.title_vocab_size),
                "pad_token_id": int(dataset.pad_token_id),
                "d_model": int(d_model),
                "n_heads": resolved_n_heads,
                "n_kv_heads": resolved_n_kv_heads,
                "n_layers": int(n_layers),
                "ff_mult": int(ff_mult),
                "dropout": float(model_dropout),
                "max_token_length": int(derived_max_token_length),
                "metadata_prefix_tokens": max(int(metadata_prefix_tokens), 8),
                "use_genre_metadata": bool(use_genre_metadata),
                "use_year_metadata": bool(use_year_metadata),
                "use_mood_metadata": bool(use_mood_metadata),
                "use_key_metadata": bool(use_key_metadata),
                "use_bpm_metadata": bool(use_bpm_metadata),
                "reference_track_duration_sec": float(reference_track_duration_sec),
                "rope_base": float(rope_base),
            }

            base_model = MetadataTokenTransformer(
                token_vocab_size=int(model_config["token_vocab_size"]),
                artist_vocab_size=int(model_config["artist_vocab_size"]),
                genre_vocab_size=int(model_config["genre_vocab_size"]),
                mood_vocab_size=int(model_config["mood_vocab_size"]),
                key_vocab_size=int(model_config["key_vocab_size"]),
                title_vocab_size=int(model_config["title_vocab_size"]),
                pad_token_id=int(model_config["pad_token_id"]),
                d_model=int(model_config["d_model"]),
                n_heads=int(model_config["n_heads"]),
                n_kv_heads=int(model_config["n_kv_heads"]),
                n_layers=int(model_config["n_layers"]),
                ff_mult=int(model_config["ff_mult"]),
                dropout=float(model_config["dropout"]),
                max_token_length=int(model_config["max_token_length"]),
                metadata_prefix_tokens=int(model_config["metadata_prefix_tokens"]),
                use_genre_metadata=bool(model_config["use_genre_metadata"]),
                use_year_metadata=bool(model_config["use_year_metadata"]),
                use_mood_metadata=bool(model_config["use_mood_metadata"]),
                use_key_metadata=bool(model_config["use_key_metadata"]),
                use_bpm_metadata=bool(model_config["use_bpm_metadata"]),
                rope_base=float(model_config["rope_base"]),
            )

            base_model = base_model.to(primary_device)

            if len(resolved_gpu_ids) > 1:
                model = torch.nn.DataParallel(
                    base_model,
                    device_ids=resolved_gpu_ids,
                    output_device=resolved_gpu_ids[0],
                )
            else:
                model = base_model

            optimizer = AdamW(
                model.parameters(),
                lr=float(learning_rate),
                weight_decay=float(weight_decay),
            )

            total_optimizer_steps = max(1, int(epochs) * max(1, len(train_loader)))
            scheduler_package = build_warmup_cosine_scheduler(
                optimizer=optimizer,
                total_steps=total_optimizer_steps,
                warmup_ratio=float(self.training_defaults.get("lr_warmup_ratio", 0.05)),
                min_lr_ratio=float(self.training_defaults.get("lr_min_ratio", 0.1)),
            )

            use_ema = bool(self.training_defaults.get("use_ema", True))
            ema_decay = float(self.training_defaults.get("ema_decay", 0.999))
            top_k_checkpoint_average = int(self.training_defaults.get("top_k_checkpoint_average", 3))

            trainer = TokenTrainer(
                model=model,
                optimizer=optimizer,
                device=primary_device,
                mixed_precision=bool(mixed_precision),
                max_grad_norm=float(self.training_defaults.get("max_grad_norm", 1.0)),
                scheduler=scheduler_package["scheduler"],
                use_ema=use_ema,
                ema_decay=ema_decay,
                top_k_checkpoint_average=top_k_checkpoint_average,
            )

            result = trainer.fit(
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=int(epochs),
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )

            saved_model = model.module if isinstance(model, torch.nn.DataParallel) else model
            parameter_count = count_model_parameters(saved_model)
            output_file_name = build_model_file_name(parameter_count)
            output_path = model_output_folder / output_file_name

            save_training_payload(
                output_path=output_path,
                saved_model=saved_model,
                model_config=model_config,
                codec_info=codec_info,
                total_len=total_len,
                train_len=train_len,
                val_len=val_len,
                length_info=length_info,
                processed_training_data_folder=processed_training_data_folder,
                resolved_gpu_ids=resolved_gpu_ids,
                primary_device=primary_device,
                batch_size=batch_size,
                epochs=epochs,
                learning_rate=learning_rate,
                validation_split=validation_split,
                weight_decay=weight_decay,
                mixed_precision=mixed_precision,
                effective_pin_memory=effective_pin_memory,
                use_genre_metadata=use_genre_metadata,
                use_year_metadata=use_year_metadata,
                use_mood_metadata=use_mood_metadata,
                use_key_metadata=use_key_metadata,
                use_bpm_metadata=use_bpm_metadata,
                parameter_count=parameter_count,
                output_file_name=output_file_name,
                metadata_vocab_path=metadata_vocab_path,
                token_manifest_path=token_manifest_path,
                history=result["history"],
                token_budget_per_batch=token_budget_per_batch,
                max_examples_per_batch=max_examples_per_batch,
                lr_scheduler_name=str(scheduler_package["name"]),
                lr_scheduler_total_steps=int(scheduler_package["total_steps"]),
                lr_warmup_ratio=float(scheduler_package["warmup_ratio"]),
                lr_warmup_steps=int(scheduler_package["warmup_steps"]),
                lr_min_ratio=float(scheduler_package["min_lr_ratio"]),
                use_ema=use_ema,
                ema_decay=ema_decay,
                top_k_checkpoint_average=top_k_checkpoint_average,
                final_selection=str(result["final_selection"]),
                averaged_checkpoint_count=int(result["averaged_checkpoint_count"]),
            )

            return {
                "final_model_path": str(output_path),
                "best_epoch": int(result["best_epoch"]),
                "best_metric": float(result["best_metric"]),
            }
        finally:
            del optimizer
            del model
            del train_loader
            del val_loader
            del dataset
            release_torch_memory()