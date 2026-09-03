import json
import os
import traceback
import webbrowser
from contextlib import suppress
from pathlib import Path

from modules.util import path_util
from modules.util.config.SecretsConfig import SecretsConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.enum.ModelType import ModelType
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.path_util import write_json_atomic


_REPO_ROOT = Path(__file__).resolve().parents[2]


class TopBarController:
    def __init__(self, config: TrainConfig):
        self.train_config = config

    def get_model_types(self) -> list[tuple[str, ModelType]]:
        return [  #TODO simplify
            ("SD1.5", ModelType.STABLE_DIFFUSION_15),
            ("SD1.5 Inpainting", ModelType.STABLE_DIFFUSION_15_INPAINTING),
            ("SD2.0", ModelType.STABLE_DIFFUSION_20),
            ("SD2.0 Inpainting", ModelType.STABLE_DIFFUSION_20_INPAINTING),
            ("SD2.1", ModelType.STABLE_DIFFUSION_21),
            ("SD3", ModelType.STABLE_DIFFUSION_3),
            ("SD3.5", ModelType.STABLE_DIFFUSION_35),
            ("SDXL", ModelType.STABLE_DIFFUSION_XL_10_BASE),
            ("SDXL Inpainting", ModelType.STABLE_DIFFUSION_XL_10_BASE_INPAINTING),
            ("Wuerstchen v2", ModelType.WUERSTCHEN_2),
            ("Stable Cascade", ModelType.STABLE_CASCADE_1),
            ("PixArt Alpha", ModelType.PIXART_ALPHA),
            ("PixArt Sigma", ModelType.PIXART_SIGMA),
            ("Flux Dev.1", ModelType.FLUX_DEV_1),
            ("Flux Fill Dev", ModelType.FLUX_FILL_DEV_1),
            ("Flux 2 [Dev, Klein]", ModelType.FLUX_2),
            ("Mage Flow", ModelType.MAGE_FLOW),
            ("Sana", ModelType.SANA),
            ("Hunyuan Video", ModelType.HUNYUAN_VIDEO),
            ("HiDream Full", ModelType.HI_DREAM_FULL),
            ("Chroma1", ModelType.CHROMA_1),
            ("QwenImage", ModelType.QWEN),
            ("Anima", ModelType.ANIMA),
            ("Krea 2", ModelType.KREA_2),
            ("PRX Pixel", ModelType.PRX_PIXEL),
            ("Z-Image", ModelType.Z_IMAGE),
            ("Ernie Image", ModelType.ERNIE),
            ("Ideogram 4", ModelType.IDEOGRAM_4),
        ]

    def get_training_methods(self, model_type: ModelType) -> list[tuple[str, TrainingMethod]]:
        labels = {
            TrainingMethod.FINE_TUNE: "Fine Tune",
            TrainingMethod.LORA: "LoRA",
            TrainingMethod.EMBEDDING: "Embedding",
            TrainingMethod.FINE_TUNE_VAE: "Fine Tune VAE",
        }
        return [(labels[m], m) for m in model_type.supported_training_methods()]

    def load_preset_tree(self, dir: str | os.PathLike[str] = "training_presets") -> list[tuple[str, str | list]]:
        # Built-in presets belong to the OneTrainer checkout, not to whichever
        # working directory happened to launch the UI. Resolve relative preset
        # roots against the repository so external launchers/VNC/Pixi wrappers
        # cannot silently build an empty or stale menu.
        directory = Path(dir)
        if not directory.is_absolute():
            directory = _REPO_ROOT / directory

        # Mirrors the directory structure under `directory`; a node is either
        # (display_name, path) for a leaf preset or (display_name, children) for
        # a group. "#" marks a built-in preset; "#.json" is last-session state.
        nodes = []
        if directory.is_dir():
            for entry in sorted(os.scandir(directory), key=lambda e: e.name.lower()):
                if entry.is_dir():
                    children = self.load_preset_tree(entry.path)
                    if children:
                        nodes.append((entry.name, children))
                elif entry.name.startswith("#") and entry.name != "#.json" and entry.name.endswith(".json"):
                    nodes.append((os.path.splitext(entry.name)[0], str(Path(entry.path))))
        return nodes

    def save_to_file(self, name) -> str:
        name = path_util.safe_filename(name)
        path = path_util.canonical_join("training_presets", f"{name}.json")
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))
        return path

    def save_config_to_path(self, path: str) -> None:
        write_json_atomic(path, self.train_config.to_settings_dict(secrets=False))

    def load_config_from_file(self, filename: str) -> TrainConfig | None:
        try:
            basename = os.path.basename(filename)
            is_built_in_preset = basename.startswith("#") and basename != "#.json"

            with open(filename, "r") as f:
                loaded_dict = json.load(f)
                default_config = TrainConfig.default_values()
                # built-in configs are always saved in the most recent version, so migration can be skipped
                loaded_config = default_config.from_dict(loaded_dict, migrate=not is_built_in_preset).to_unpacked_config()

            with suppress(FileNotFoundError), open("secrets.json", "r") as f:
                secrets_dict = json.load(f)
                loaded_config.secrets = SecretsConfig.default_values().from_dict(secrets_dict)

            self.train_config.from_dict(loaded_config.to_dict())
            return loaded_config
        except FileNotFoundError:
            return None
        except Exception:
            print(traceback.format_exc())
            return None

    def save_secrets(self, path) -> str:
        secrets = self.train_config.secrets.to_dict()
        # Dataset keys are session-only. Saving them beside the configuration
        # would defeat the purpose of encrypting data before uploading it.
        secrets["dataset_encryption_key"] = ""
        write_json_atomic(path, secrets)
        return path

    def open_wiki(self):
        webbrowser.open("https://github.com/Nerogar/OneTrainer/wiki", new=0, autoraise=False)

    def save_default(self):
        self.save_to_file("#")
        self.save_secrets("secrets.json")
