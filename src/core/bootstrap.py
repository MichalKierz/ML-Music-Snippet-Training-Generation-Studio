import shutil
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from src.core.config_loader import load_all_configs
from src.core.device_utils import get_default_preprocess_device, get_default_training_gpu_ids
from src.core.logger import configure_logging, get_logger
from src.core.paths import ensure_runtime_directories, get_base_dir, get_user_settings_path
from src.core.runtime_context import RuntimeContext
from src.core.settings_manager import SettingsManager


def build_default_user_settings(runtime_dirs: dict, configs: dict) -> dict:
    training_defaults = configs["training"]
    preprocess_defaults = configs["preprocess"]
    generate_defaults = configs["generate"]
    token_model_defaults = configs["token_model"]
    app_defaults = configs["app"]

    return {
        "paths": {
            "raw_music_folder": str(runtime_dirs["training_input"]),
            "processed_training_data_folder": str(runtime_dirs["processed_training_data"]),
            "model_output_folder": str(runtime_dirs["models"]),
            "generate_output_folder": str(runtime_dirs["generated"]),
            "selected_model_file": "",
        },
        "hardware": {
            "preprocess_device": get_default_preprocess_device(),
            "training_gpu_ids": get_default_training_gpu_ids(),
        },
        "training": {
            "chunk_length_sec": float(preprocess_defaults["chunk_length_sec"]),
            "chunk_stride_sec": float(preprocess_defaults["chunk_stride_sec"]),
            "batch_size": int(training_defaults["batch_size"]),
            "batch_token_budget": int(training_defaults.get("batch_token_budget", 0)),
            "use_token_bucket_batching": bool(training_defaults.get("use_token_bucket_batching", True)),
            "bucket_size_multiplier": int(training_defaults.get("bucket_size_multiplier", 50)),
            "epochs": int(training_defaults["epochs"]),
            "learning_rate": float(training_defaults["learning_rate"]),
            "lr_warmup_ratio": float(training_defaults.get("lr_warmup_ratio", 0.03)),
            "lr_min_ratio": float(training_defaults.get("lr_min_ratio", 0.1)),
            "validation_split": float(training_defaults["validation_split"]),
            "weight_decay": float(training_defaults["weight_decay"]),
            "mixed_precision": bool(training_defaults["mixed_precision"]),
            "pin_memory": bool(training_defaults.get("pin_memory", False)),
            "use_ema": bool(training_defaults.get("use_ema", True)),
            "ema_decay": float(training_defaults.get("ema_decay", 0.999)),
            "use_genre_metadata": bool(training_defaults.get("use_genre_metadata", token_model_defaults.get("use_genre_metadata", True))),
            "use_year_metadata": bool(training_defaults.get("use_year_metadata", token_model_defaults.get("use_year_metadata", True))),
            "use_mood_metadata": bool(training_defaults.get("use_mood_metadata", token_model_defaults.get("use_mood_metadata", False))),
            "use_key_metadata": bool(training_defaults.get("use_key_metadata", token_model_defaults.get("use_key_metadata", False))),
            "use_bpm_metadata": bool(training_defaults.get("use_bpm_metadata", token_model_defaults.get("use_bpm_metadata", False))),
        },
        "token_model": {
            "d_model": int(token_model_defaults["d_model"]),
            "n_heads": int(token_model_defaults["n_heads"]),
            "n_kv_heads": int(token_model_defaults.get("n_kv_heads", token_model_defaults["n_heads"])),
            "n_layers": int(token_model_defaults["n_layers"]),
            "ff_mult": int(token_model_defaults["ff_mult"]),
            "dropout": float(token_model_defaults["dropout"]),
            "metadata_prefix_tokens": int(max(8, int(token_model_defaults["metadata_prefix_tokens"]))),
            "reference_track_duration_sec": float(token_model_defaults["reference_track_duration_sec"]),
            "rope_base": float(token_model_defaults.get("rope_base", 10000.0)),
        },
        "generate": {
            "artist": "",
            "title": "",
            "year": 0,
            "genre": "",
            "mood": "",
            "initial_key": "",
            "bpm": 0.0,
            "position_mode": str(generate_defaults["position_mode"]),
            "start_time": str(generate_defaults["start_time"]),
            "relative_position": float(generate_defaults["relative_position"]),
            "section": str(generate_defaults["section"]),
            "temperature": float(generate_defaults.get("temperature", 1.0)),
            "top_k": int(generate_defaults.get("top_k", 50)),
            "seed": "",
            "generate_song": bool(generate_defaults.get("generate_song", False)),
            "song_snippet_count": int(generate_defaults.get("song_snippet_count", 50)),
        },
        "window": {
            "width": int(app_defaults["window"]["width"]),
            "height": int(app_defaults["window"]["height"]),
            "last_tab_index": 0,
        },
    }


def migrate_processed_training_data_folder(base_dir: Path, runtime_dirs: dict, logger):
    old_dir = base_dir / "preprocessed_data"
    new_dir = runtime_dirs["processed_training_data"]

    if old_dir.exists() and old_dir.resolve() != new_dir.resolve():
        if not new_dir.exists():
            try:
                shutil.move(str(old_dir), str(new_dir))
                logger.info(f"Migrated processed training data folder from {old_dir} to {new_dir}")
            except Exception as exc:
                logger.warning(f"Could not migrate processed training data folder from {old_dir} to {new_dir}: {exc}")


def _merge_defaults(existing: dict, defaults: dict) -> dict:
    result = dict(existing)
    for key, value in defaults.items():
        if key not in result:
            result[key] = value
    return result


def migrate_user_settings(settings: SettingsManager, base_dir: Path, runtime_dirs: dict, configs: dict, logger):
    old_dir = str((base_dir / "preprocessed_data").resolve())
    new_dir = str(runtime_dirs["processed_training_data"].resolve())

    paths_section = dict(settings.get("paths"))
    hardware_section = dict(settings.get("hardware"))
    training_section = dict(settings.get("training"))
    token_model_section = dict(settings.get("token_model"))
    generate_section = dict(settings.get("generate"))
    window_section = dict(settings.get("window"))

    if "processed_training_data_folder" not in paths_section or not paths_section.get("processed_training_data_folder"):
        paths_section["processed_training_data_folder"] = new_dir

    if str(Path(paths_section.get("processed_training_data_folder", new_dir)).resolve()) == old_dir:
        paths_section["processed_training_data_folder"] = new_dir

    if "raw_music_folder" not in paths_section:
        paths_section["raw_music_folder"] = str(runtime_dirs["training_input"])

    if "model_output_folder" not in paths_section:
        paths_section["model_output_folder"] = str(runtime_dirs["models"])

    if "generate_output_folder" not in paths_section:
        paths_section["generate_output_folder"] = str(runtime_dirs["generated"])

    if "selected_model_file" not in paths_section:
        paths_section["selected_model_file"] = ""

    paths_section.pop("preprocessed_data_folder", None)

    default_settings = build_default_user_settings(runtime_dirs, configs)

    hardware_section = _merge_defaults(hardware_section, default_settings["hardware"])
    training_section = _merge_defaults(training_section, default_settings["training"])
    token_model_section = _merge_defaults(token_model_section, default_settings["token_model"])
    generate_section = _merge_defaults(generate_section, default_settings["generate"])
    window_section = _merge_defaults(window_section, default_settings["window"])

    legacy_training_keys = {
        "sample_rate",
        "n_mels",
        "latent_dim",
        "early_stopping",
        "device_preference",
        "use_artist_metadata",
        "use_title_metadata",
        "use_position_metadata",
        "use_duration_metadata",
    }

    for key in legacy_training_keys:
        training_section.pop(key, None)

    token_model_section.pop("max_token_length", None)
    token_model_section["metadata_prefix_tokens"] = int(max(8, int(token_model_section.get("metadata_prefix_tokens", 8))))
    token_model_section["n_kv_heads"] = int(token_model_section.get("n_kv_heads", token_model_section.get("n_heads", 8)))
    token_model_section["rope_base"] = float(token_model_section.get("rope_base", 10000.0))

    training_section["batch_token_budget"] = int(training_section.get("batch_token_budget", 0))
    training_section["use_token_bucket_batching"] = bool(training_section.get("use_token_bucket_batching", True))
    training_section["bucket_size_multiplier"] = int(training_section.get("bucket_size_multiplier", 50))
    training_section["lr_warmup_ratio"] = float(training_section.get("lr_warmup_ratio", 0.03))
    training_section["lr_min_ratio"] = float(training_section.get("lr_min_ratio", 0.1))
    training_section["use_ema"] = bool(training_section.get("use_ema", True))
    training_section["ema_decay"] = float(training_section.get("ema_decay", 0.999))

    settings.set_section("paths", paths_section)
    settings.set_section("hardware", hardware_section)
    settings.set_section("training", training_section)
    settings.set_section("token_model", token_model_section)
    settings.set_section("generate", generate_section)
    settings.set_section("window", window_section)

    logger.info("User settings migrated to decoder-only token pipeline")


def create_runtime_context() -> RuntimeContext:
    base_dir = get_base_dir()
    configure_logging(base_dir)
    logger = get_logger(__name__)
    logger.info("Creating runtime context")

    configs = load_all_configs(base_dir / "configs")
    runtime_dirs = ensure_runtime_directories(base_dir, configs["app"])

    migrate_processed_training_data_folder(base_dir, runtime_dirs, logger)

    user_settings_path = get_user_settings_path(base_dir)
    default_user_settings = build_default_user_settings(runtime_dirs, configs)
    settings = SettingsManager(user_settings_path, default_user_settings)

    migrate_user_settings(settings, base_dir, runtime_dirs, configs, logger)

    logger.info("Runtime context ready")
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"User settings file: {user_settings_path}")

    return RuntimeContext(
        base_dir=base_dir,
        configs=configs,
        runtime_dirs=runtime_dirs,
        settings=settings,
    )


def run_app():
    from src.ui.main_window import MainWindow

    context = create_runtime_context()
    logger = get_logger(__name__)
    logger.info("Starting Qt application")

    app = QApplication(sys.argv)
    window = MainWindow(context)
    window.show()

    exit_code = app.exec()
    logger.info(f"Application exited with code {exit_code}")
    sys.exit(exit_code)