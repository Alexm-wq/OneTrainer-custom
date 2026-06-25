import contextlib
import copy
import json
import math
import os
import shutil
import traceback
from collections.abc import Callable
from pathlib import Path

import modules.util.multi_gpu_util as multi
from modules.dataLoader.BaseDataLoader import BaseDataLoader
from modules.model.BaseModel import BaseModel
from modules.modelLoader.BaseModelLoader import BaseModelLoader
from modules.modelSampler.BaseModelSampler import BaseModelSampler, ModelSamplerOutput
from modules.modelSaver.BaseModelSaver import BaseModelSaver
from modules.modelSetup.BaseModelSetup import BaseModelSetup
from modules.trainer.BaseTrainer import BaseTrainer
from modules.util import create, path_util
from modules.util.bf16_stochastic_rounding import set_seed as bf16_stochastic_rounding_set_seed
from modules.util.callbacks.TrainCallbacks import TrainCallbacks
from modules.util.commands.TrainCommands import TrainCommands
from modules.util.compile_util import init_compile
from modules.util.config.SampleConfig import SampleConfig
from modules.util.config.TrainConfig import TrainConfig
from modules.util.dpo_beta_controller import DPOBetaController
from modules.util.dtype_util import create_grad_scaler, enable_grad_scaling
from modules.util.enum.ConceptType import ConceptType
from modules.util.enum.DPOObjective import DPOObjective
from modules.util.enum.EMAMode import EMAMode
from modules.util.enum.FileType import FileType
from modules.util.enum.ModelFormat import ModelFormat
from modules.util.enum.TimeUnit import TimeUnit
from modules.util.enum.TrainingMethod import TrainingMethod
from modules.util.profiling_util import TorchMemoryRecorder, TorchProfiler
from modules.util.time_util import get_string_timestamp
from modules.util.torch_util import torch_gc
from modules.util.TrainProgress import TrainProgress

import torch
from torch import Tensor, nn
from torch.nn import Parameter
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import pil_to_tensor

import huggingface_hub
from requests.exceptions import ConnectionError
from tqdm import tqdm


class GenericTrainer(BaseTrainer):
    model_loader: BaseModelLoader
    model_setup: BaseModelSetup
    data_loader: BaseDataLoader
    model_saver: BaseModelSaver
    model_sampler: BaseModelSampler
    model: BaseModel | None
    validation_data_loader: BaseDataLoader

    previous_sample_time: float
    sample_queue: list[Callable]

    parameters: list[Parameter]

    tensorboard: SummaryWriter

    grad_hook_handles: list[RemovableHandle]

    def __init__(self, config: TrainConfig, callbacks: TrainCallbacks, commands: TrainCommands):
        super().__init__(config, callbacks, commands)
        # torch._dynamo.config overrides are thread-local, so init_compile() must be called in the training thread/process.
        init_compile()

        if multi.is_master():
            tensorboard_log_dir = os.path.join(config.workspace_dir, "tensorboard")
            os.makedirs(Path(tensorboard_log_dir).absolute(), exist_ok=True)
            self.tensorboard = SummaryWriter(os.path.join(tensorboard_log_dir, f"{config.save_filename_prefix}{get_string_timestamp()}"))
            if config.tensorboard and not config.tensorboard_always_on:
                super()._start_tensorboard()

        self.model = None
        self.one_step_trained = False
        self.grad_hook_handles = []

    def start(self):
        if multi.is_master():
            self.__save_config_to_workspace()

            if self.config.clear_cache_before_training and self.config.latent_caching:
                self.__clear_cache()

        if self.config.train_dtype.enable_tf():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model_loader = self.create_model_loader()
        self.model_setup = self.create_model_setup()

        self.callbacks.on_update_status("loading the model")

        model_names = self.config.model_names()

        if self.config.continue_last_backup:
            self.callbacks.on_update_status("searching for previous backups")
            last_backup_path = self.config.get_last_backup_path()

            if last_backup_path:
                if self.config.training_method == TrainingMethod.LORA:
                    model_names.lora = last_backup_path
                elif self.config.training_method == TrainingMethod.EMBEDDING:
                    model_names.embedding.model_name = last_backup_path
                else:  # fine-tunes
                    model_names.base_model = last_backup_path

                print(f"Continuing training from backup '{last_backup_path}'...")
            else:
                print("No backup found, continuing without backup...")

        if self.config.secrets.huggingface_token != "":
            self.callbacks.on_update_status("logging into Hugging Face")
            with contextlib.suppress(ConnectionError):
                huggingface_hub.login(token=self.config.secrets.huggingface_token)

        self.callbacks.on_update_status("loading the model")

        if self.config.quantization.cache_dir is None:
            self.config.quantization.cache_dir = self.config.cache_dir + "/quantization"
        os.makedirs(self.config.quantization.cache_dir, exist_ok=True)

        self.model = self.model_loader.load(
            model_type=self.config.model_type,
            model_names=model_names,
            weight_dtypes=self.config.weight_dtypes(),
            quantization=self.config.quantization,
        )
        self.model.train_config = self.config

        self.callbacks.on_update_status("running model setup")

        self.model_setup.setup_optimizations(self.model, self.config)
        self.model_setup.setup_train_device(self.model, self.config)
        self.model_setup.setup_model(self.model, self.config)
        self.model.to(self.temp_device)
        self.model.eval()
        torch_gc()

        if self.config.rlhf_enabled and self.config.training_method != TrainingMethod.LORA:
            raise NotImplementedError("RLHF DPO is currently implemented for adapter training in the LoRA tab only.")

        self.callbacks.on_update_status("creating the data loader/caching")

        self.data_loader = self.create_data_loader(
            self.model, self.model_setup, self.model.train_progress
        )
        self.model_saver = self.create_model_saver()

        self.model_sampler = self.create_model_sampler(self.model)
        self.previous_sample_time = -1
        self.sample_queue = []

        self.parameters = self.model.parameters.parameters()

        if self.config.validation or self.config.rlhf_dpo_validation:
            self.validation_data_loader = self.create_data_loader(
                self.model, self.model_setup, self.model.train_progress, is_validation=True
            )

        self._dpo_patience_counter = 0
        self._dpo_best_accuracy = float("-inf")
        self._dpo_best_margin = float("-inf")
        self._dpo_best_backup_path: str | None = None
        self._dpo_beta_controller: DPOBetaController | None = None

    def __save_config_to_workspace(self):
        path = path_util.canonical_join(self.config.workspace_dir, "config")
        os.makedirs(Path(path).absolute(), exist_ok=True)
        path = path_util.canonical_join(path, f"{self.config.save_filename_prefix}{get_string_timestamp()}.json")
        with open(path, "w") as f:
            json.dump(self.config.to_pack_dict(secrets=False), f, indent=4)

    def __clear_cache(self):
        print(
            f'Clearing cache directory {self.config.cache_dir}! '
            f'You can disable this if you want to continue using the same cache.'
        )
        if os.path.isdir(self.config.cache_dir):
            for filename in os.listdir(self.config.cache_dir):
                path = os.path.join(self.config.cache_dir, filename)
                if os.path.isdir(path) and (filename.startswith('epoch-') or filename in ['image', 'text'] or filename.startswith('image-rlhf-') or filename.startswith('text-rlhf-')):
                    shutil.rmtree(path)

    def __prune_backups(self, backups_to_keep: int):
        backup_dirpath = os.path.join(self.config.workspace_dir, "backup")
        if os.path.exists(backup_dirpath):
            backup_directories = sorted(
                [dirpath for dirpath in os.listdir(backup_dirpath) if
                 os.path.isdir(os.path.join(backup_dirpath, dirpath))],
                reverse=True,
            )

            for dirpath in backup_directories[backups_to_keep:]:
                dirpath = os.path.join(backup_dirpath, dirpath)
                try:
                    shutil.rmtree(dirpath)
                except Exception:
                    print(f"Could not delete old rolling backup {dirpath}")

        return

    def __enqueue_sample_during_training(self, fun: Callable):
        self.sample_queue.append(fun)

    def __execute_sample_during_training(self):
        for fun in self.sample_queue:
            fun()
        self.sample_queue = []

    def __sample_loop(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_config_list: list[SampleConfig],
            ema_applied: bool,
            folder_postfix: str = "",
            is_custom_sample: bool = False,
    ):
        for i, sample_config in multi.distributed(
            [(i, sample_config) for i, sample_config in enumerate(sample_config_list) if sample_config.enabled],
            distribute=not self.config.samples_to_tensorboard and not ema_applied
        ):
            try:
                safe_prompt = path_util.safe_filename(sample_config.prompt)

                if is_custom_sample:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        "custom",
                    )
                else:
                    sample_dir = os.path.join(
                        self.config.workspace_dir,
                        "samples",
                        f"{str(i)} - {safe_prompt}{folder_postfix}",
                    )

                sample_path = os.path.join(
                    sample_dir,
                    f"{self.config.save_filename_prefix}{get_string_timestamp()}-training-sample-{train_progress.filename_string()}"
                )

                def on_sample_default(sampler_output: ModelSamplerOutput):
                    if self.config.samples_to_tensorboard and sampler_output.file_type == FileType.IMAGE:
                        self.tensorboard.add_image(
                            f"sample{str(i)} - {safe_prompt}", pil_to_tensor(sampler_output.data),  # noqa: B023
                            train_progress.global_step
                        )
                    self.callbacks.on_sample_default(sampler_output)

                def on_sample_custom(sampler_output: ModelSamplerOutput):
                    self.callbacks.on_sample_custom(sampler_output)

                on_sample = on_sample_custom if is_custom_sample else on_sample_default
                on_update_progress = self.callbacks.on_update_sample_custom_progress if is_custom_sample else self.callbacks.on_update_sample_default_progress

                self.model.to(self.temp_device)
                self.model.eval()

                sample_config = copy.copy(sample_config)
                sample_config.from_train_config(self.config)

                self.model_sampler.sample(
                    sample_config=sample_config,
                    destination=sample_path,
                    image_format=self.config.sample_image_format,
                    video_format=self.config.sample_video_format,
                    audio_format=self.config.sample_audio_format,
                    on_sample=on_sample,
                    on_update_progress=on_update_progress,
                )
            except Exception:
                traceback.print_exc()
                print("Error during sampling, proceeding without sampling")

            torch_gc()

    def __sample_during_training(
            self,
            train_progress: TrainProgress,
            train_device: torch.device,
            sample_params_list: list[SampleConfig] = None,
    ):
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()
        torch_gc()

        self.callbacks.on_update_status("Sampling ...")

        is_custom_sample = False
        if sample_params_list:
            is_custom_sample = True
        elif self.config.samples is not None:
            sample_params_list = self.config.samples
        else:
            try:
                with open(self.config.sample_definition_file_name, 'r') as f:
                    samples = json.load(f)
                    for i in range(len(samples)):
                        samples[i] = SampleConfig.default_values(self.config.model_type).from_dict(samples[i])
                    sample_params_list = samples
            # We absolutely do not want to fail training just because the sample definition file becomes missing or broken right before sampling.
            except Exception:
                traceback.print_exc()
                print("Error during loading the sample definition file, proceeding without sampling")
                sample_params_list = []

        if self.model.ema:
            #the EMA model only exists in the master process, so EMA sampling is done on one GPU only
            #non-EMA sampling is done on all GPUs
            assert multi.is_master() and self.config.ema != EMAMode.OFF
            self.model.ema.copy_ema_to(self.parameters, store_temp=True)

        self.__sample_loop(
            train_progress=train_progress,
            train_device=train_device,
            sample_config_list=sample_params_list,
            is_custom_sample=is_custom_sample,
            ema_applied = self.config.ema != EMAMode.OFF
        )

        if self.model.ema:
            self.model.ema.copy_temp_to(self.parameters)

        # ema-less sampling, if ema is enabled:
        if self.config.ema != EMAMode.OFF and not is_custom_sample and self.config.non_ema_sampling:
            self.__sample_loop(
                train_progress=train_progress,
                train_device=train_device,
                sample_config_list=sample_params_list,
                folder_postfix=" - no-ema",
                ema_applied = False,
            )

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __validate(self, train_progress: TrainProgress):
        if self.__needs_validate(train_progress):
            self.validation_data_loader.get_data_set().start_next_epoch()
            current_epoch_length_validation = self.validation_data_loader.get_data_set().approximate_length()

            if current_epoch_length_validation == 0:
                return

            self.callbacks.on_update_status("Calculating validation loss")
            self.model_setup.setup_train_device(self.model, self.config)

            torch_gc()

            step_tqdm_validation = tqdm(
                self.validation_data_loader.get_data_loader(),
                desc="validation_step",
                total=current_epoch_length_validation)

            if self.config.rlhf_dpo_validation:
                dpo_val_accuracy = []
                dpo_val_chosen_reward = []
                dpo_val_rejected_reward = []
                dpo_val_reward_margin = []

                for validation_batch in step_tqdm_validation:
                    if self.__needs_gc(train_progress):
                        torch_gc()

                    dpo_indices = self.__rlhf_dpo_indices(validation_batch)
                    if not dpo_indices:
                        continue
                    validation_batch = self.__subbatch(validation_batch, dpo_indices)
                    with torch.no_grad():
                        self.model_setup.calculate_dpo_loss(self.model, validation_batch, self.config, train_progress)
                        self.__log_dpo_chosen_crash(batch, self.model_setup.get_last_dpo_metrics(), train_progress)
                    dpo_metrics = self.model_setup.get_last_dpo_metrics()
                    dpo_val_accuracy.append(dpo_metrics["accuracy"])
                    dpo_val_chosen_reward.append(dpo_metrics["chosen_reward"])
                    dpo_val_rejected_reward.append(dpo_metrics["rejected_reward"])
                    dpo_val_reward_margin.append(dpo_metrics["reward_margin"])

                if dpo_val_accuracy:
                    # Validation loss is intentionally not tracked: reward hacking
                    # drives it toward zero, so a low val loss is not evidence of a
                    # good model. Held-out ranking accuracy is hack-resistant.
                    val_accuracy = sum(dpo_val_accuracy) / len(dpo_val_accuracy)
                    val_chosen_reward = sum(dpo_val_chosen_reward) / len(dpo_val_chosen_reward)
                    val_rejected_reward = sum(dpo_val_rejected_reward) / len(dpo_val_rejected_reward)
                    val_reward_margin = sum(dpo_val_reward_margin) / len(dpo_val_reward_margin)

                    self.tensorboard.add_scalar("dpo/val_accuracy", val_accuracy, train_progress.global_step)
                    self.tensorboard.add_scalar("dpo/val_chosen_reward", val_chosen_reward, train_progress.global_step)
                    self.tensorboard.add_scalar(
                        "dpo/val_rejected_reward", val_rejected_reward, train_progress.global_step
                    )
                    self.tensorboard.add_scalar("dpo/val_reward_margin", val_reward_margin, train_progress.global_step)
                    self.__check_dpo_patience(val_accuracy, val_reward_margin, train_progress)

                # DPO validation uses a different data pipeline (paired samples) than
                # standard validation, so they cannot share the same data loader.
                return

            accumulated_loss_per_concept = {}
            concept_counts = {}
            mapping_seed_to_label = {}
            mapping_label_to_seed = {}

            for validation_batch in step_tqdm_validation:
                if self.__needs_gc(train_progress):
                    torch_gc()

                with torch.no_grad():
                    model_output_data = self.model_setup.predict(
                        self.model, validation_batch, self.config, train_progress, deterministic=True)
                    loss_validation = self.model_setup.calculate_loss(
                        self.model, validation_batch, model_output_data, self.config)

                concept_name = validation_batch["concept_name"][0]
                concept_path = validation_batch["concept_path"][0]
                concept_seed = validation_batch["concept_seed"].item()
                loss = loss_validation.item()

                label = concept_name if concept_name else os.path.basename(concept_path)
                if label in mapping_label_to_seed and mapping_label_to_seed[label] != concept_seed:
                    suffix = 1
                    new_label = f"{label}({suffix})"
                    while new_label in mapping_label_to_seed and mapping_label_to_seed[new_label] != concept_seed:
                        suffix += 1
                        new_label = f"{label}({suffix})"
                    label = new_label

                if concept_seed not in mapping_seed_to_label:
                    mapping_seed_to_label[concept_seed] = label
                    mapping_label_to_seed[label] = concept_seed

                accumulated_loss_per_concept[concept_seed] = accumulated_loss_per_concept.get(concept_seed, 0) + loss
                concept_counts[concept_seed] = concept_counts.get(concept_seed, 0) + 1

            for concept_seed, total_loss in accumulated_loss_per_concept.items():
                average_loss = total_loss / concept_counts[concept_seed]
                label = mapping_seed_to_label[concept_seed]

                self.tensorboard.add_scalar(f"loss/validation_step/{label}", average_loss, train_progress.global_step)

            if len(concept_counts) > 1:
                total_loss = sum(accumulated_loss_per_concept[key] for key in concept_counts)
                total_count = sum(concept_counts[key] for key in concept_counts)
                total_average_loss = total_loss / total_count

                self.tensorboard.add_scalar("loss/validation_step/total_average",
                                            total_average_loss,
                                            train_progress.global_step)

    def __check_dpo_patience(self, val_accuracy: float, val_reward_margin: float, train_progress: TrainProgress):
        rounded_accuracy = round(val_accuracy, 5)
        rounded_best_accuracy = round(self._dpo_best_accuracy, 5)

        is_new_best = rounded_accuracy > rounded_best_accuracy or (
            rounded_accuracy == rounded_best_accuracy and val_reward_margin > self._dpo_best_margin
        )

        if is_new_best and self.config.rlhf_dpo_save_best:
            self._dpo_best_backup_path = self.__save_dpo_best(val_accuracy, val_reward_margin, train_progress)

        if is_new_best:
            self._dpo_best_accuracy = val_accuracy
            self._dpo_best_margin = val_reward_margin

        if not self.config.rlhf_dpo_patience_enabled:
            return

        if is_new_best:
            self._dpo_patience_counter = 0
        else:
            self._dpo_patience_counter += 1

        self.tensorboard.add_scalar("dpo/patience_counter", self._dpo_patience_counter, train_progress.global_step)

        if self._dpo_patience_counter >= self.config.rlhf_dpo_patience_value:
            print(
                f"DPO early stopping triggered: patience exhausted after {self._dpo_patience_counter} "
                f"consecutive checks without improvement."
            )
            self.commands.stop()

    def __save_dpo_best(self, val_accuracy: float, val_reward_margin: float, train_progress: TrainProgress) -> str:
        best_path = os.path.join(self.config.workspace_dir, "backup", "dpo-best.pt")
        os.makedirs(os.path.dirname(best_path), exist_ok=True)
        try:
            state = [p.data.clone().cpu() for p in self.parameters]
            torch.save(state, best_path)
            print(
                f"Saved DPO best checkpoint (accuracy={val_accuracy:.4f}, "
                f"margin={val_reward_margin:.6f}) to {best_path}"
            )
        except Exception:
            traceback.print_exc()
            print("Could not save DPO best checkpoint.")
            return self._dpo_best_backup_path or ""
        return best_path

    def __save_backup_config(self, backup_path):
        config_path = os.path.join(backup_path, "onetrainer_config")
        args_path = path_util.canonical_join(config_path, "args.json")
        concepts_path = path_util.canonical_join(config_path, "concepts.json")
        samples_path = path_util.canonical_join(config_path, "samples.json")

        os.makedirs(Path(config_path).absolute(), exist_ok=True)

        with open(args_path, "w") as f:
            json.dump(self.config.to_settings_dict(secrets=False), f, indent=4)
        if os.path.isfile(self.config.concept_file_name):
            shutil.copy2(self.config.concept_file_name, concepts_path)
        if os.path.isfile(self.config.sample_definition_file_name):
            shutil.copy2(self.config.sample_definition_file_name, samples_path)

    def __backup(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Creating backup")

        backup_name = f"{get_string_timestamp()}-backup-{train_progress.filename_string()}"
        backup_path = os.path.join(self.config.workspace_dir, "backup", backup_name)

        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()

        try:
            if print_msg:
                print_cb("Creating Backup " + backup_path)

            self.model_saver.save(
                self.model,
                self.config.model_type,
                ModelFormat.INTERNAL,
                backup_path,
                None,
            )

            self.__save_backup_config(backup_path)

            # AVG_LOSS_TENSORBOARD_PATCH: save average-loss state with this backup.
            _avg_loss_state_path = os.path.join(backup_path, "onetrainer_config", "avg_loss_state.json")
            with open(_avg_loss_state_path, "w") as _avg_loss_state_file:
                json.dump(
                    {
                        "epoch": train_progress.epoch,
                        "epoch_step": getattr(train_progress, "epoch_step", None),
                        "global_step": train_progress.global_step,
                        "avg_loss_total": getattr(self, "_avg_loss_total", 0.0),
                        "avg_loss_steps": getattr(self, "_avg_loss_steps", 0),
                        "epoch_loss_total": getattr(self, "_avg_loss_epoch_total", 0.0),
                        "epoch_loss_steps": getattr(self, "_avg_loss_epoch_steps", 0),
                    },
                    _avg_loss_state_file,
                    indent=4,
                )
        except Exception:
            traceback.print_exc()
            print("Could not save backup. Check your disk space!")
            try:
                if os.path.isdir(backup_path):
                    shutil.rmtree(backup_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial backup")
        finally:
            if self.config.rolling_backup:
                self.__prune_backups(self.config.rolling_backup_count)

        self.model_setup.setup_train_device(self.model, self.config)
        # Special case for schedule-free optimizers.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.train()

        torch_gc()

    def __save(self, train_progress: TrainProgress, print_msg: bool = True, print_cb: Callable[[str], None] = print):
        torch_gc()

        self.callbacks.on_update_status("Saving")

        save_path = os.path.join(
            self.config.workspace_dir,
            "save",
            f"{self.config.save_filename_prefix}{get_string_timestamp()}-save-{train_progress.filename_string()}{self.config.output_model_format.file_extension()}"
        )
        if print_msg:
            print_cb("Saving " + save_path)

        try:
            if self.model.ema:
                self.model.ema.copy_ema_to(self.parameters, store_temp=True)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()
            self.model_saver.save(
                model=self.model,
                model_type=self.config.model_type,
                output_model_format=self.config.output_model_format,
                output_model_destination=save_path,
                dtype=self.config.output_dtype.torch_dtype()
            )
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()
        except Exception:
            traceback.print_exc()
            print("Could not save model. Check your disk space!")
            try:
                if os.path.isfile(save_path):
                    shutil.rmtree(save_path)
            except Exception:
                traceback.print_exc()
                print("Could not delete partial save")
        finally:
            if self.model.ema:
                self.model.ema.copy_temp_to(self.parameters)

        torch_gc()

    def __needs_sample(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "sample_skip_first", self.config.sample_skip_first, self.config.sample_after_unit, train_progress
        ) and self.repeating_action_needed(
            "sample", self.config.sample_after, self.config.sample_after_unit, train_progress
        )

    def __needs_backup(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "backup", self.config.backup_after, self.config.backup_after_unit, train_progress, start_at_zero=False
        )

    def __needs_save(self, train_progress: TrainProgress):
        return self.single_action_elapsed(
            "save_skip_first", self.config.save_skip_first, self.config.save_every_unit, train_progress
        ) and self.repeating_action_needed(
            "save", self.config.save_every, self.config.save_every_unit, train_progress, start_at_zero=False
        )

    def __needs_gc(self, train_progress: TrainProgress):
        return self.repeating_action_needed("gc", 5, TimeUnit.MINUTE, train_progress, start_at_zero=False)

    def __needs_validate(self, train_progress: TrainProgress):
        return self.repeating_action_needed(
            "validate", self.config.validate_after, self.config.validate_after_unit, train_progress
        )

    def __is_update_step(self, train_progress: TrainProgress) -> bool:
        return self.repeating_action_needed(
            "update_step", self.config.gradient_accumulation_steps, TimeUnit.STEP, train_progress, start_at_zero=False
        )

    def __apply_fused_back_pass(self, scaler):
        fused_optimizer_step = self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass
        fused_reduce = self.config.multi_gpu and self.config.fused_gradient_reduce
        if fused_optimizer_step:
            if self.config.gradient_accumulation_steps > 1:
                print("Warning: activating Fused Back Pass with Accumulation Steps > 1 does not reduce VRAM usage.")
            if self.config.multi_gpu and not fused_reduce:
                raise ValueError("if Fused Back Pass and Multi-GPU is enabled, Fused Reduce must also be enabled")
        elif not fused_reduce:
            return

        for param_group in self.model.optimizer.param_groups:
            for i, parameter in enumerate(param_group["params"]):
                # TODO: Find a better check instead of "parameter.requires_grad".
                #       This will break if the some parameters don't require grad during the first training step.
                if parameter.requires_grad:
                    if scaler:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            scaler.unscale_parameter_(tensor, self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            scaler.maybe_opt_step_parameter(tensor, param_group, i, self.model.optimizer)
                            tensor.grad = None
                    else:
                        def __optimizer_step(tensor: Tensor, param_group=param_group, i=i):
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(tensor, self.config.clip_grad_norm)
                            self.model.optimizer.step_parameter(tensor, param_group, i)
                            tensor.grad = None

                    def __grad_hook(tensor: Tensor, param_group=param_group, i=i):
                        init_compile()  # workaround for https://github.com/pytorch/pytorch/issues/186537
                        if self.__is_update_step(self.model.train_progress):
                            if fused_reduce:
                                multi.reduce_grads_mean(
                                    [tensor],
                                    self.config.gradient_reduce_precision,
                                    after_reduce=__optimizer_step if fused_optimizer_step else None,
                                    async_op=self.config.async_gradient_reduce,
                                    max_buffer=self.config.async_gradient_reduce_buffer * 1024 * 1024,
                                )
                            elif fused_optimizer_step:
                                __optimizer_step(tensor)

                    handle = parameter.register_post_accumulate_grad_hook(__grad_hook)
                    self.grad_hook_handles.append(handle)


    def __before_eval(self):
        # Special case for schedule-free optimizers, which need eval()
        # called before evaluation. Can and should move this to a callback
        # during a refactoring.
        if self.config.optimizer.optimizer.is_schedule_free:
            torch.clear_autocast_cache()
            self.model.optimizer.eval()


    def __batch_len(self, batch: dict) -> int:
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim > 0:
                return int(value.shape[0])
            if isinstance(value, (list, tuple)):
                return len(value)
        return int(self.config.batch_size)

    @staticmethod
    def __as_bool(value) -> bool:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return False
            value = value.detach().cpu().flatten()[0].item()
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y", "dpo", "paired"}
        return bool(value)

    def __rlhf_dpo_indices(self, batch: dict) -> list[int]:
        batch_len = self.__batch_len(batch)

        if "dpo_is_paired" not in batch:
            if self.config.rlhf_enabled and "latent_image_rejected" in batch:
                return list(range(batch_len))
            return []

        flags = batch["dpo_is_paired"]
        if isinstance(flags, torch.Tensor):
            values = flags.detach().cpu().flatten().tolist()
        elif isinstance(flags, (list, tuple)):
            values = list(flags)
        else:
            values = [flags] * batch_len

        return [i for i, value in enumerate(values[:batch_len]) if self.__as_bool(value)]

    def __normal_indices(self, batch: dict) -> list[int]:
        dpo = set(self.__rlhf_dpo_indices(batch))
        return [i for i in range(self.__batch_len(batch)) if i not in dpo]

    def __subbatch(self, batch: dict, indices: list[int]) -> dict:
        batch_len = self.__batch_len(batch)
        out = {}
        tensor_indices = None

        for key, value in batch.items():
            if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == batch_len:
                if tensor_indices is None:
                    tensor_indices = torch.tensor(indices, device=value.device, dtype=torch.long)
                out[key] = value.index_select(0, tensor_indices)
            elif isinstance(value, list) and len(value) == batch_len:
                out[key] = [value[i] for i in indices]
            elif isinstance(value, tuple) and len(value) == batch_len:
                out[key] = tuple(value[i] for i in indices)
            else:
                out[key] = value

        return out

    def __concept_type_at(self, batch: dict, index: int) -> ConceptType:
        raw = batch["concept_type"][index]
        if isinstance(raw, torch.Tensor):
            raw = raw.detach().cpu().item()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="ignore")
        if isinstance(raw, ConceptType):
            return raw
        return ConceptType(raw)

    def __calculate_standard_training_loss(
        self,
        batch: dict,
        train_progress: TrainProgress,
    ) -> Tensor:
        batch_len = self.__batch_len(batch)

        prior_pred_indices = [
            i for i in range(batch_len)
            if self.__concept_type_at(batch, i) == ConceptType.PRIOR_PREDICTION
        ]

        if len(prior_pred_indices) > 0 or (
            self.config.masked_training
            and self.config.masked_prior_preservation_weight > 0
            and self.config.training_method == TrainingMethod.LORA
        ):
            with self.model_setup.prior_model(self.model, self.config), torch.no_grad():
                prior_model_output_data = self.model_setup.predict(
                    self.model,
                    batch,
                    self.config,
                    train_progress,
                )

            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)
            prior_model_prediction = prior_model_output_data["predicted"].to(
                dtype=model_output_data["target"].dtype
            )
            model_output_data["target"][prior_pred_indices] = prior_model_prediction[prior_pred_indices]
            model_output_data["prior_target"] = prior_model_prediction
        else:
            model_output_data = self.model_setup.predict(self.model, batch, self.config, train_progress)

        return self.model_setup.calculate_loss(self.model, batch, model_output_data, self.config)


    # DPO_CHOSEN_CRASH_HELPER_PATCH
    def __log_dpo_chosen_crash(self, batch, dpo_metrics, train_progress):
        """
        Chosen-only DPO crash logger.

        Logs only when chosen_reward suddenly goes hard negative or drops hard
        from its own EMA. Rejected reward is logged only as context.
        """
        try:
            import csv as _csv
            import math as _math
            from pathlib import Path as _Path

            if not dpo_metrics:
                return

            chosen = float(dpo_metrics.get("chosen_reward", 0.0))
            rejected = float(dpo_metrics.get("rejected_reward", 0.0))
            margin = float(dpo_metrics.get("reward_margin", chosen - rejected))
            dpo_loss = float(dpo_metrics.get("dpo_loss", dpo_metrics.get("loss", 0.0)))
            accuracy = float(dpo_metrics.get("accuracy", 0.0))

            bad_numeric = (
                _math.isnan(chosen) or _math.isinf(chosen)
                or _math.isnan(dpo_loss) or _math.isinf(dpo_loss)
            )

            if not hasattr(self, "_dpo_chosen_crash_steps"):
                self._dpo_chosen_crash_steps = 0
                self._dpo_chosen_reward_ema = None
                self._dpo_chosen_reward_var = 1e-12

            decay = 0.99
            warmup = 10

            # Edit these if it is too noisy or too quiet.
            abs_threshold = -0.020
            drop_threshold = 0.015
            z_threshold = -6.0

            if self._dpo_chosen_reward_ema is None:
                self._dpo_chosen_reward_ema = chosen
                self._dpo_chosen_reward_var = 1e-12
                chosen_z = 0.0
            else:
                std = max(self._dpo_chosen_reward_var, 1e-12) ** 0.5
                chosen_z = (chosen - self._dpo_chosen_reward_ema) / std

            chosen_drop = self._dpo_chosen_reward_ema - chosen

            absolute_crash = chosen <= abs_threshold
            ema_crash = (
                self._dpo_chosen_crash_steps >= warmup
                and chosen_drop >= drop_threshold
            )
            z_crash = (
                self._dpo_chosen_crash_steps >= warmup
                and chosen_z <= z_threshold
            )

            is_crash = bad_numeric or absolute_crash or ema_crash or z_crash

            def batch_value(key):
                v = ""
                try:
                    if isinstance(batch, dict):
                        v = batch.get(key, "")
                    else:
                        v = getattr(batch, key, "")
                except Exception:
                    v = ""

                if isinstance(v, (list, tuple)):
                    return " | ".join(str(x) for x in v[:32])

                return str(v)

            def batch_len(key):
                try:
                    v = batch.get(key, "") if isinstance(batch, dict) else getattr(batch, key, "")
                    if isinstance(v, (list, tuple)):
                        return len(v)
                except Exception:
                    pass
                return ""

            if is_crash:
                reasons = []
                if bad_numeric:
                    reasons.append("nan_or_inf")
                if absolute_crash:
                    reasons.append("chosen_absolute_negative")
                if ema_crash:
                    reasons.append("chosen_drop_from_ema")
                if z_crash:
                    reasons.append("chosen_negative_z")

                # Prefer original source paths if the DPO source-path patch is installed.
                # Fall back to raw image_path if not.
                chosen_source = batch_value("dpo_chosen_source_path") or batch_value("image_path")
                rejected_source_context = batch_value("dpo_rejected_source_path") or batch_value("image_path_rejected")

                workspace_dir = getattr(self.config, "workspace_dir", None)
                if workspace_dir is None:
                    workspace_dir = getattr(self.config, "workspace", ".")

                csv_path = _Path(workspace_dir) / "dpo_chosen_reward_crashes.csv"
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not csv_path.exists()

                with csv_path.open("a", encoding="utf-8", newline="") as f:
                    fieldnames = [
                        "global_step",
                        "epoch",
                        "epoch_step",
                        "reason",
                        "chosen_reward",
                        "chosen_reward_ema",
                        "chosen_reward_drop",
                        "chosen_reward_z",
                        "rejected_reward_context",
                        "reward_margin",
                        "dpo_loss",
                        "accuracy",
                        "batch_image_count",
                        "chosen_source_path",
                        "chosen_image_path_raw",
                        "rejected_source_path_context",
                        "dpo_pair_key",
                        "concept_name",
                        "concept_path",
                        "prompt",
                    ]

                    writer = _csv.DictWriter(f, fieldnames=fieldnames)
                    if write_header:
                        writer.writeheader()

                    writer.writerow({
                        "global_step": getattr(train_progress, "global_step", ""),
                        "epoch": getattr(train_progress, "epoch", ""),
                        "epoch_step": getattr(train_progress, "epoch_step", ""),
                        "reason": "+".join(reasons),
                        "chosen_reward": chosen,
                        "chosen_reward_ema": self._dpo_chosen_reward_ema,
                        "chosen_reward_drop": chosen_drop,
                        "chosen_reward_z": chosen_z,
                        "rejected_reward_context": rejected,
                        "reward_margin": margin,
                        "dpo_loss": dpo_loss,
                        "accuracy": accuracy,
                        "batch_image_count": batch_len("image_path"),
                        "chosen_source_path": chosen_source,
                        "chosen_image_path_raw": batch_value("image_path"),
                        "rejected_source_path_context": rejected_source_context,
                        "dpo_pair_key": batch_value("dpo_pair_key"),
                        "concept_name": batch_value("concept_name"),
                        "concept_path": batch_value("concept_path"),
                        "prompt": batch_value("text"),
                    })

                try:
                    self.tensorboard.add_scalar(
                        "dpo/chosen_crash_flag",
                        1.0,
                        getattr(train_progress, "global_step", 0),
                    )
                    self.tensorboard.add_scalar(
                        "dpo/chosen_crash_reward",
                        chosen,
                        getattr(train_progress, "global_step", 0),
                    )
                except Exception:
                    pass

                print(
                    f"[DPO CHOSEN CRASH] "
                    f"step={getattr(train_progress, 'global_step', '')} "
                    f"reason={'+'.join(reasons)} "
                    f"chosen={chosen:.6g} "
                    f"ema={self._dpo_chosen_reward_ema:.6g} "
                    f"drop={chosen_drop:.6g} "
                    f"z={chosen_z:.3f} "
                    f"path={chosen_source} "
                    f"csv={csv_path}"
                )

            # Update EMA after judging this batch against the previous trend.
            if not bad_numeric:
                diff = chosen - self._dpo_chosen_reward_ema
                self._dpo_chosen_reward_ema = (
                    self._dpo_chosen_reward_ema * decay
                    + chosen * (1.0 - decay)
                )
                self._dpo_chosen_reward_var = (
                    self._dpo_chosen_reward_var * decay
                    + (diff * diff) * (1.0 - decay)
                )
                self._dpo_chosen_crash_steps += 1

        except Exception as e:
            print(f"[DPO CHOSEN CRASH LOGGER ERROR] {e}")



    # DPO_PAIR_LOSS_STATE_LOG_PATCH
    def __log_dpo_pair_loss_state(self, batch, dpo_metrics, train_progress):
        """
        Logs every DPO microbatch/pair with raw loss and saturation state.

        saturated means:
            dpo_loss <= 1e-3

        With batch size 1 this gives the exact pair.
        With batch size >1 it logs the paths in that DPO microbatch.
        """
        try:
            import csv as _csv
            from pathlib import Path as _Path

            if not dpo_metrics:
                return

            dpo_loss = float(dpo_metrics.get("dpo_loss", dpo_metrics.get("loss", 0.0)))
            chosen_reward = float(dpo_metrics.get("chosen_reward", 0.0))
            rejected_reward = float(dpo_metrics.get("rejected_reward", 0.0))
            reward_margin = float(dpo_metrics.get("reward_margin", chosen_reward - rejected_reward))
            accuracy = float(dpo_metrics.get("accuracy", 0.0))

            saturation_threshold = 1e-3
            is_saturated = dpo_loss <= saturation_threshold

            def batch_value(key):
                try:
                    v = batch.get(key, "") if isinstance(batch, dict) else getattr(batch, key, "")
                except Exception:
                    v = ""

                if isinstance(v, (list, tuple)):
                    return " | ".join(str(x) for x in v[:64])

                return str(v)

            def batch_len(key):
                try:
                    v = batch.get(key, "") if isinstance(batch, dict) else getattr(batch, key, "")
                    if isinstance(v, (list, tuple)):
                        return len(v)
                except Exception:
                    pass
                return ""

            chosen_source = batch_value("dpo_chosen_source_path") or batch_value("image_path")
            rejected_source = batch_value("dpo_rejected_source_path") or batch_value("image_path_rejected")

            workspace_dir = getattr(self.config, "workspace_dir", None)
            if workspace_dir is None:
                workspace_dir = getattr(self.config, "workspace", ".")

            csv_path = _Path(workspace_dir) / "dpo_pair_loss_states.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)

            write_header = not csv_path.exists()

            with csv_path.open("a", encoding="utf-8", newline="") as f:
                fieldnames = [
                    "global_step",
                    "epoch",
                    "epoch_step",
                    "dpo_loss",
                    "saturation_threshold",
                    "is_saturated",
                    "chosen_reward",
                    "rejected_reward",
                    "reward_margin",
                    "accuracy",
                    "batch_image_count",
                    "chosen_source_path",
                    "rejected_source_path",
                    "chosen_image_path_raw",
                    "rejected_image_path_raw",
                    "dpo_pair_key",
                    "concept_name",
                    "concept_path",
                    "prompt",
                ]

                writer = _csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()

                writer.writerow({
                    "global_step": getattr(train_progress, "global_step", ""),
                    "epoch": getattr(train_progress, "epoch", ""),
                    "epoch_step": getattr(train_progress, "epoch_step", ""),
                    "dpo_loss": dpo_loss,
                    "saturation_threshold": saturation_threshold,
                    "is_saturated": is_saturated,
                    "chosen_reward": chosen_reward,
                    "rejected_reward": rejected_reward,
                    "reward_margin": reward_margin,
                    "accuracy": accuracy,
                    "batch_image_count": batch_len("image_path"),
                    "chosen_source_path": chosen_source,
                    "rejected_source_path": rejected_source,
                    "chosen_image_path_raw": batch_value("image_path"),
                    "rejected_image_path_raw": batch_value("image_path_rejected"),
                    "dpo_pair_key": batch_value("dpo_pair_key"),
                    "concept_name": batch_value("concept_name"),
                    "concept_path": batch_value("concept_path"),
                    "prompt": batch_value("text"),
                })

            try:
                self.tensorboard.add_scalar(
                    "dpo/pair_is_saturated",
                    1.0 if is_saturated else 0.0,
                    getattr(train_progress, "global_step", 0),
                )
                self.tensorboard.add_scalar(
                    "dpo/pair_loss_logged",
                    dpo_loss,
                    getattr(train_progress, "global_step", 0),
                )
            except Exception:
                pass

        except Exception as e:
            print(f"[DPO PAIR LOSS LOGGER ERROR] {e}")


    def __calculate_mixed_rlhf_training_loss(
        self,
        batch: dict,
        train_progress: TrainProgress,
    ) -> tuple[Tensor, dict[str, float] | None]:
        if not self.config.rlhf_enabled:
            return self.__calculate_standard_training_loss(batch, train_progress), None

        dpo_indices = self.__rlhf_dpo_indices(batch)
        if not dpo_indices:
            return self.__calculate_standard_training_loss(batch, train_progress), None

        normal_indices = self.__normal_indices(batch)

        if not normal_indices:
            dpo_loss = self.model_setup.calculate_dpo_loss(self.model, batch, self.config, train_progress)
            self.__log_dpo_chosen_crash(batch, self.model_setup.get_last_dpo_metrics(), train_progress)
            return dpo_loss, self.model_setup.get_last_dpo_metrics()

        total_items = len(dpo_indices) + len(normal_indices)
        loss_parts: list[Tensor] = []
        dpo_metrics: dict[str, float] | None = None

        if normal_indices:
            normal_batch = self.__subbatch(batch, normal_indices)
            normal_loss = self.__calculate_standard_training_loss(normal_batch, train_progress)
            loss_parts.append(normal_loss * (len(normal_indices) / total_items))

        if dpo_indices:
            dpo_batch = self.__subbatch(batch, dpo_indices)
            dpo_loss = self.model_setup.calculate_dpo_loss(self.model, dpo_batch, self.config, train_progress)
            self.__log_dpo_chosen_crash(dpo_batch, self.model_setup.get_last_dpo_metrics(), train_progress)
            loss_parts.append(dpo_loss * (len(dpo_indices) / total_items))
            dpo_metrics = self.model_setup.get_last_dpo_metrics()

        return sum(loss_parts), dpo_metrics

    def train(self):
        train_device = torch.device(self.config.train_device)

        train_progress = self.model.train_progress

        if self.config.only_cache:
            if multi.is_master():
                self.callbacks.on_update_status("Caching")
                for _epoch in tqdm(range(train_progress.epoch, self.config.epochs, 1), desc="epoch"):
                    self.data_loader.get_data_set().start_next_epoch()
            return

        scaler = create_grad_scaler() if enable_grad_scaling(self.config.train_dtype, self.parameters) else None

        self.__apply_fused_back_pass(scaler)

        # False if the model gradients are all None, True otherwise
        # This is used to schedule sampling only when the gradients don't take up any space
        has_gradient = False

        lr_scheduler = None
        accumulated_loss = torch.tensor(0.0, device=train_device)
        accumulated_dpo_metrics: dict[str, float] | None = None
        ema_loss = None
        ema_loss_steps = 0

        # AVG_LOSS_TENSORBOARD_PATCH: average loss counters.
        avg_loss_total = 0.0
        avg_loss_steps = 0
        epoch_loss_total = 0.0
        epoch_loss_steps = 0
        self._avg_loss_total = avg_loss_total
        self._avg_loss_steps = avg_loss_steps
        self._avg_loss_epoch_total = epoch_loss_total
        self._avg_loss_epoch_steps = epoch_loss_steps
        ema_reward_margin = None
        ema_reward_margin_steps = 0
        ema_chosen_reward = None
        ema_dpo_accuracy = None
        reward_hacking_streak = 0
        epochs = range(train_progress.epoch, self.config.epochs, 1)

        for _epoch in tqdm(epochs, desc="epoch") if multi.is_master() else epochs:
            multi.sync_commands(self.commands)
            if self.commands.get_stop_command():
                return
            self.callbacks.on_update_status("Starting epoch/caching")

            #call start_next_epoch with only one process at first, because it might write to the cache. All subsequent processes can read in parallel:
            for _ in multi.master_first():
                if self.config.latent_caching:
                    self.data_loader.get_data_set().start_next_epoch()
                    self.model_setup.setup_train_device(self.model, self.config)
                else:
                    self.model_setup.setup_train_device(self.model, self.config)
                    self.data_loader.get_data_set().start_next_epoch()

            if self.config.debug_mode:
                multi.warn_parameter_divergence(self.parameters, train_device)

            # Special case for schedule-free optimizers, which need train()
            # called before training. Can and should move this to a callback
            # during a refactoring.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.train()

            torch_gc()

            if lr_scheduler is None:
                lr_scheduler = create.create_lr_scheduler(
                    config=self.config,
                    optimizer=self.model.optimizer,
                    learning_rate_scheduler=self.config.learning_rate_scheduler,
                    warmup_steps=self.config.learning_rate_warmup_steps,
                    num_cycles=self.config.learning_rate_cycles,
                    min_factor=self.config.learning_rate_min_factor,
                    num_epochs=self.config.epochs,
                    approximate_epoch_length=self.data_loader.get_data_set().approximate_length(),
                    batch_size=self.config.batch_size,
                    gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                    global_step=train_progress.global_step
                )

            current_epoch_length = self.data_loader.get_data_set().approximate_length()

            # AVG_LOSS_TENSORBOARD_PATCH: reset current epoch average loss.
            epoch_loss_total = 0.0
            epoch_loss_steps = 0
            self._avg_loss_epoch_total = epoch_loss_total
            self._avg_loss_epoch_steps = epoch_loss_steps

            if getattr(self.config, "continue_last_backup", False):
                try:
                    _avg_loss_backup_path = self.config.get_last_backup_path()
                    if _avg_loss_backup_path:
                        _avg_loss_state_path = os.path.join(_avg_loss_backup_path, "onetrainer_config", "avg_loss_state.json")
                        if os.path.isfile(_avg_loss_state_path):
                            with open(_avg_loss_state_path, "r") as _avg_loss_state_file:
                                _avg_loss_state = json.load(_avg_loss_state_file)
                            if (
                                _avg_loss_state.get("epoch") == train_progress.epoch
                                and _avg_loss_state.get("epoch_step") == getattr(train_progress, "epoch_step", None)
                                and _avg_loss_state.get("global_step") == train_progress.global_step
                            ):
                                avg_loss_total = float(_avg_loss_state.get("avg_loss_total", avg_loss_total))
                                avg_loss_steps = int(_avg_loss_state.get("avg_loss_steps", avg_loss_steps))
                                epoch_loss_total = float(_avg_loss_state.get("epoch_loss_total", epoch_loss_total))
                                epoch_loss_steps = int(_avg_loss_state.get("epoch_loss_steps", epoch_loss_steps))
                                self._avg_loss_total = avg_loss_total
                                self._avg_loss_steps = avg_loss_steps
                                self._avg_loss_epoch_total = epoch_loss_total
                                self._avg_loss_epoch_steps = epoch_loss_steps
                except Exception:
                    pass

            if multi.is_master():
                batches = step_tqdm = tqdm(self.data_loader.get_data_loader(), desc="step", total=current_epoch_length,
                                 initial=train_progress.epoch_step)
            else:
                batches = self.data_loader.get_data_loader()
            for batch in batches:
                multi.sync_commands(self.commands)
                if self.commands.get_stop_command():
                    multi.warn_parameter_divergence(self.parameters, train_device)

                if not self.commands.get_stop_command() and self.__needs_sample(train_progress) or self.commands.get_and_reset_sample_default_command():
                    self.__enqueue_sample_during_training(
                        lambda: self.__sample_during_training(train_progress, train_device)
                    )
                if self.__needs_backup(train_progress):
                    self.commands.backup()

                if self.__needs_save(train_progress):
                    self.commands.save()

                sample_commands = self.commands.get_and_reset_sample_custom_commands()
                if sample_commands:
                    def create_sample_commands_fun(sample_commands):
                        def sample_commands_fun():
                            self.__sample_during_training(train_progress, train_device, sample_commands)

                        return sample_commands_fun

                    self.__enqueue_sample_during_training(create_sample_commands_fun(sample_commands))

                if self.__needs_gc(train_progress):
                    torch_gc()

                if not has_gradient:
                    self.__execute_sample_during_training()
                    backup = self.commands.get_and_reset_backup_command()
                    save = self.commands.get_and_reset_save_command()
                    if multi.is_master() and (backup or save):
                        self.model.to(self.temp_device)
                        if backup:
                            self.__backup(train_progress, True, step_tqdm.write)
                        if save:
                            self.__save(train_progress, True, step_tqdm.write)
                        self.model_setup.setup_train_device(self.model, self.config)

                self.callbacks.on_update_status("Training ...")

                with (
                    TorchMemoryRecorder(enabled=False, filename=f"memory-step{train_progress.global_step}.pickle"),
                    TorchProfiler      (enabled=False, filename=f"profile-step{train_progress.global_step}.json"),
                ):
                    step_seed = train_progress.global_step
                    bf16_stochastic_rounding_set_seed(step_seed, train_device)

                    loss, micro_dpo_metrics = self.__calculate_mixed_rlhf_training_loss(
                        batch,
                        train_progress,
                    )
                    if micro_dpo_metrics:
                        self.__log_dpo_pair_loss_state(batch, micro_dpo_metrics, train_progress)
                        if accumulated_dpo_metrics is None:
                            accumulated_dpo_metrics = dict.fromkeys(micro_dpo_metrics, 0.0)
                            accumulated_dpo_metrics["_count"] = 0
                        for _k, _v in micro_dpo_metrics.items():
                            accumulated_dpo_metrics[_k] += _v
                        accumulated_dpo_metrics["_count"] += 1
                    loss = loss / self.config.gradient_accumulation_steps
                    if scaler:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    has_gradient = True
                    detached_loss = loss.detach()
                    multi.reduce_tensor_mean(detached_loss)
                    accumulated_loss += detached_loss

                    if self.__is_update_step(train_progress):
                        if self.config.fused_gradient_reduce:
                            multi.finish_async(self.config.gradient_reduce_precision)
                        else:
                            multi.reduce_grads_mean(self.parameters, self.config.gradient_reduce_precision)

                        if scaler and self.config.optimizer.optimizer.supports_fused_back_pass() and self.config.optimizer.fused_back_pass:
                            scaler.step_after_unscale_parameter_(self.model.optimizer)
                            scaler.update()
                        elif scaler:
                            scaler.unscale_(self.model.optimizer)
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            scaler.step(self.model.optimizer)
                            scaler.update()
                        else:
                            if self.config.clip_grad_norm is not None:
                                nn.utils.clip_grad_norm_(self.parameters, self.config.clip_grad_norm)
                            self.model.optimizer.step()

                        lr_scheduler.step()  # done before zero_grad, because some lr schedulers need gradients
                        self.model.optimizer.zero_grad(set_to_none=True)
                        has_gradient = False

                        if multi.is_master():
                            self.model_setup.report_to_tensorboard(
                                self.model, self.config, lr_scheduler, self.tensorboard
                            )

                            accumulated_loss_cpu = accumulated_loss.item()

                            # AVG_LOSS_TENSORBOARD_PATCH: whole-run and current-epoch average loss.
                            avg_loss_total += accumulated_loss_cpu
                            avg_loss_steps += 1
                            epoch_loss_total += accumulated_loss_cpu
                            epoch_loss_steps += 1
                            self._avg_loss_total = avg_loss_total
                            self._avg_loss_steps = avg_loss_steps
                            self._avg_loss_epoch_total = epoch_loss_total
                            self._avg_loss_epoch_steps = epoch_loss_steps
                            if multi.is_master():
                                self.tensorboard.add_scalar("avg_loss/train_step", avg_loss_total / avg_loss_steps, train_progress.global_step)
                            if math.isnan(accumulated_loss_cpu):
                                raise RuntimeError("Training loss became NaN. This may be due to invalid parameters, precision issues, or a bug in the loss computation.")

                            self.tensorboard.add_scalar(
                                "loss/train_step", accumulated_loss_cpu, train_progress.global_step
                            )
                            if self.config.rlhf_enabled and accumulated_dpo_metrics is not None:
                                count = accumulated_dpo_metrics.pop("_count")
                                dpo_metrics = {k: v / count for k, v in accumulated_dpo_metrics.items()}
                                self.tensorboard.add_scalar("loss/dpo", dpo_metrics["loss"], train_progress.global_step)
                                self.tensorboard.add_scalar(
                                    "dpo/raw_loss", dpo_metrics["dpo_loss"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/chosen_reward", dpo_metrics["chosen_reward"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/rejected_reward", dpo_metrics["rejected_reward"], train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/reward_margin", dpo_metrics["reward_margin"], train_progress.global_step
                                )
                                ema_reward_margin = ema_reward_margin or dpo_metrics["reward_margin"]
                                ema_reward_margin_steps += 1
                                ema_reward_margin_decay = min(0.99, 1 - (1 / ema_reward_margin_steps))
                                ema_reward_margin = (ema_reward_margin * ema_reward_margin_decay) + (
                                    dpo_metrics["reward_margin"] * (1 - ema_reward_margin_decay)
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/smooth_reward_margin", ema_reward_margin, train_progress.global_step
                                )
                                self.tensorboard.add_scalar(
                                    "dpo/accuracy", dpo_metrics["accuracy"], train_progress.global_step
                                )

                                # Reward-hacking signature: the margin keeps growing while
                                # BOTH rewards go negative (the model degrades chosen and
                                # rejected alike) and held-out ranking saturates.
                                ema_chosen_reward = ema_chosen_reward or dpo_metrics["chosen_reward"]
                                ema_chosen_reward = (ema_chosen_reward * ema_reward_margin_decay) + (
                                    dpo_metrics["chosen_reward"] * (1 - ema_reward_margin_decay)
                                )
                                ema_dpo_accuracy = ema_dpo_accuracy or dpo_metrics["accuracy"]
                                ema_dpo_accuracy = (ema_dpo_accuracy * ema_reward_margin_decay) + (
                                    dpo_metrics["accuracy"] * (1 - ema_reward_margin_decay)
                                )
                                if ema_chosen_reward < 0 and ema_dpo_accuracy > 0.95:
                                    reward_hacking_streak += 1
                                else:
                                    reward_hacking_streak = 0
                                if reward_hacking_streak == 25:
                                    warning = (
                                        "DPO reward-hacking signature detected: chosen reward has stayed "
                                        "negative while training accuracy is saturated. The margin is likely "
                                        "growing by degrading both images of each pair. Lower the learning "
                                        "rate, raise beta, or switch to the IPO objective."
                                    )
                                    print(warning)
                                    self.tensorboard.add_text("dpo/warnings", warning, train_progress.global_step)

                                if (
                                    self.config.rlhf_dpo_adaptive_beta
                                    and self.config.rlhf_dpo_objective == DPOObjective.SIGMOID
                                ):
                                    if self._dpo_beta_controller is None:
                                        self._dpo_beta_controller = DPOBetaController(self.config.rlhf_dpo_beta)
                                    adaptive_beta = self._dpo_beta_controller.update(dpo_metrics["reward_margin"])
                                    self.model_setup.set_dpo_runtime_beta(adaptive_beta)
                                    self.tensorboard.add_scalar("dpo/beta", adaptive_beta, train_progress.global_step)
                                    self.tensorboard.add_scalar(
                                        "dpo/raw_margin_ema",
                                        self._dpo_beta_controller.margin_ema,
                                        train_progress.global_step,
                                    )

                                if self.config.rlhf_dpo_timestep_margin_logging:
                                    for quartile in range(4):
                                        quartile_count = dpo_metrics.get(f"margin_t_q{quartile + 1}_count", 0.0)
                                        if quartile_count > 0:
                                            self.tensorboard.add_scalar(
                                                f"dpo/margin_by_t/q{quartile + 1}",
                                                dpo_metrics[f"margin_t_q{quartile + 1}_sum"] / quartile_count,
                                                train_progress.global_step,
                                            )
                            ema_loss = ema_loss or accumulated_loss_cpu
                            ema_loss_steps += 1
                            ema_loss_decay = min(0.99, 1 - (1 / ema_loss_steps))
                            ema_loss = (ema_loss * ema_loss_decay) + (accumulated_loss_cpu * (1 - ema_loss_decay))
                            step_tqdm.set_postfix({
                                'loss': accumulated_loss_cpu,
                                'smooth loss': ema_loss,
                            })
                            self.tensorboard.add_scalar("smooth_loss/train_step", ema_loss, train_progress.global_step)

                        accumulated_loss = 0.0
                        accumulated_dpo_metrics = None
                        self.model_setup.after_optimizer_step(self.model, self.config, train_progress)

                        if self.model.ema:
                            assert multi.is_master()
                            update_step = train_progress.global_step // self.config.gradient_accumulation_steps
                            self.tensorboard.add_scalar(
                                "ema_decay",
                                self.model.ema.get_current_decay(update_step),
                                train_progress.global_step
                            )
                            self.model.ema.step(
                                self.parameters,
                                update_step
                            )

                        self.one_step_trained = True

                if (self.config.validation or self.config.rlhf_dpo_validation) and multi.is_master():
                    self.__validate(train_progress)

                train_progress.next_step(self.config.batch_size)
                self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

                if self.commands.get_stop_command():
                    return

            # AVG_LOSS_TENSORBOARD_PATCH: final average loss for this completed epoch.
            if multi.is_master() and epoch_loss_steps > 0:
                self.tensorboard.add_scalar("avg_loss/epoch", epoch_loss_total / epoch_loss_steps, train_progress.epoch)

            train_progress.next_epoch()
            self.callbacks.on_update_train_progress(train_progress, current_epoch_length, self.config.epochs)

            if self.commands.get_stop_command():
                return

    def end(self):
        if self.one_step_trained:
            self.model.to(self.temp_device)

            if self.config.backup_before_save and multi.is_master():
                self.__backup(self.model.train_progress)

            # Special case for schedule-free optimizers.
            if self.config.optimizer.optimizer.is_schedule_free:
                torch.clear_autocast_cache()
                self.model.optimizer.eval()

            if multi.is_master():
                self.callbacks.on_update_status("Saving the final model")

                if self.model.ema:
                    self.model.ema.copy_ema_to(self.parameters, store_temp=False)

                # Restore DPO best AFTER EMA copy so it takes precedence
                if (
                    self.config.rlhf_enabled
                    and self.config.rlhf_dpo_save_best
                    and self._dpo_best_backup_path
                    and os.path.isfile(self._dpo_best_backup_path)
                ):
                    print(f"Restoring DPO best checkpoint from {self._dpo_best_backup_path}")
                    self.callbacks.on_update_status("Restoring best DPO checkpoint")
                    best_state = torch.load(self._dpo_best_backup_path, map_location=self.temp_device)
                    for param, saved in zip(self.parameters, best_state, strict=True):
                        param.data.copy_(saved)
                    del best_state
                if (
                    os.path.isdir(self.config.output_model_destination)
                    and self.config.output_model_format.is_single_file()
                ):
                    save_path = os.path.join(
                        self.config.output_model_destination,
                        f"{self.config.save_filename_prefix}{get_string_timestamp()}{self.config.output_model_format.file_extension()}"
                    )
                else:
                    save_path = self.config.output_model_destination
                print("Saving " + save_path)

                self.model_saver.save(
                    model=self.model,
                    model_type=self.config.model_type,
                    output_model_format=self.config.output_model_format,
                    output_model_destination=save_path,
                    dtype=self.config.output_dtype.torch_dtype()
                )

        if self.model is not None:
            self.model.to(self.temp_device)

        if multi.is_master():
            self.tensorboard.close()

            if self.config.tensorboard and not self.config.tensorboard_always_on:
                super()._stop_tensorboard()

        for handle in self.grad_hook_handles:
            handle.remove()
