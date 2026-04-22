import math

import torch
import torch.nn.functional as F

from src.core.task_cancelled import TaskCancelledError
from src.training.model_averaging import ExponentialMovingAverage, average_state_dicts, clone_state_dict


class TokenTrainer:
    def __init__(
        self,
        model,
        optimizer,
        device,
        mixed_precision: bool,
        max_grad_norm: float,
        scheduler=None,
        use_ema: bool = False,
        ema_decay: float = 0.999,
        top_k_checkpoint_average: int = 1,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.mixed_precision = bool(mixed_precision)
        self.max_grad_norm = float(max_grad_norm)
        self.scheduler = scheduler
        self.use_ema = bool(use_ema)
        self.ema_decay = float(ema_decay)
        self.top_k_checkpoint_average = max(1, int(top_k_checkpoint_average))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision and self.device.type == "cuda")
        self.ema = ExponentialMovingAverage(model=self.model, decay=self.ema_decay) if self.use_ema else None

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int,
        progress_callback=None,
        is_cancelled=None,
    ):
        history = []
        best_metric = math.inf
        best_epoch = 0
        best_state_dict = None
        top_checkpoint_states: list[tuple[float, int, dict[str, torch.Tensor]]] = []

        for epoch in range(1, int(epochs) + 1):
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token training cancelled.")

            if hasattr(train_loader, "batch_sampler") and hasattr(train_loader.batch_sampler, "set_epoch"):
                train_loader.batch_sampler.set_epoch(epoch - 1)

            train_metrics = self._run_epoch(
                loader=train_loader,
                training=True,
                epoch=epoch,
                total_epochs=int(epochs),
                progress_callback=progress_callback,
                is_cancelled=is_cancelled,
            )

            if self.ema is not None:
                eval_state_dict = self.ema.state_dict(to_cpu=True)
            else:
                eval_state_dict = clone_state_dict(self.model.state_dict(), to_cpu=True)

            if val_loader is not None:
                raw_state_dict = clone_state_dict(self.model.state_dict(), to_cpu=False)
                self.model.load_state_dict(eval_state_dict, strict=True)

                val_metrics = self._run_epoch(
                    loader=val_loader,
                    training=False,
                    epoch=epoch,
                    total_epochs=int(epochs),
                    progress_callback=progress_callback,
                    is_cancelled=is_cancelled,
                )
                monitored_loss = float(val_metrics["loss"])

                self.model.load_state_dict(raw_state_dict, strict=True)
            else:
                val_metrics = None
                monitored_loss = float(train_metrics["loss"])

            current_lr = float(self.optimizer.param_groups[0]["lr"])

            record = {
                "epoch": int(epoch),
                "train_loss": float(train_metrics["loss"]),
                "val_loss": None if val_metrics is None else float(val_metrics["loss"]),
                "learning_rate": current_lr,
                "selection_source": "ema" if self.ema is not None else "raw",
            }
            history.append(record)

            if monitored_loss < best_metric:
                best_metric = monitored_loss
                best_epoch = int(epoch)
                best_state_dict = clone_state_dict(eval_state_dict, to_cpu=True)

            top_checkpoint_states.append(
                (
                    float(monitored_loss),
                    int(epoch),
                    clone_state_dict(eval_state_dict, to_cpu=True),
                )
            )
            top_checkpoint_states.sort(key=lambda item: (item[0], item[1]))
            top_checkpoint_states = top_checkpoint_states[: self.top_k_checkpoint_average]

        if len(top_checkpoint_states) >= 2:
            final_state_dict = average_state_dicts([item[2] for item in top_checkpoint_states])
            final_selection = f"average_top_{len(top_checkpoint_states)}"
        elif best_state_dict is not None:
            final_state_dict = best_state_dict
            final_selection = "best_single"
        elif self.ema is not None:
            final_state_dict = self.ema.state_dict(to_cpu=True)
            final_selection = "ema_last"
        else:
            final_state_dict = clone_state_dict(self.model.state_dict(), to_cpu=True)
            final_selection = "raw_last"

        self.model.load_state_dict(final_state_dict, strict=True)

        return {
            "history": history,
            "best_metric": float(best_metric),
            "best_epoch": int(best_epoch),
            "final_selection": str(final_selection),
            "averaged_checkpoint_count": int(len(top_checkpoint_states)),
        }

    def _run_epoch(
        self,
        loader,
        training: bool,
        epoch: int,
        total_epochs: int,
        progress_callback=None,
        is_cancelled=None,
    ):
        self.model.train(training)

        total_loss = 0.0
        total_batches = 0
        batch_total = max(1, len(loader))

        for batch_index, batch in enumerate(loader, start=1):
            if is_cancelled and is_cancelled():
                raise TaskCancelledError("Token training cancelled.")

            batch = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in batch.items()
            }

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.mixed_precision and self.device.type == "cuda"):
                logits = self.model(
                    input_ids=batch["input_ids"],
                    artist_id=batch["artist_id"],
                    genre_id=batch["genre_id"],
                    mood_id=batch["mood_id"],
                    key_id=batch["key_id"],
                    title_tokens=batch["title_tokens"],
                    title_length=batch["title_length"],
                    year_value=batch["year_value"],
                    position_value=batch["position_value"],
                    bpm_value=batch["bpm_value"],
                    attention_mask=batch["attention_mask"],
                )

                loss = F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    batch["target_ids"].reshape(-1),
                    ignore_index=-100,
                )

            if training:
                self.scaler.scale(loss).backward()
                if self.max_grad_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.ema is not None:
                    self.ema.update(self.model)

            loss_value = float(loss.detach().cpu().item())
            current_lr = float(self.optimizer.param_groups[0]["lr"])

            total_loss += loss_value
            total_batches += 1

            if progress_callback is not None:
                batch_fraction = batch_index / batch_total
                epoch_progress = (epoch - 1) + batch_fraction
                phase = "Training" if training else "Validation"
                progress_callback(
                    "Token Training",
                    float(epoch_progress),
                    float(total_epochs),
                    f"Epoch {epoch}/{total_epochs} | {phase} batch {batch_index}/{batch_total} | loss={loss_value:.6f} | lr={current_lr:.8f}",
                )

        if total_batches == 0:
            return {"loss": 0.0}

        return {"loss": total_loss / total_batches}