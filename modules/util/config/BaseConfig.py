from collections.abc import Callable
from enum import Enum
from typing import Any, get_args, get_origin

from modules.util.type_util import issubclass_safe


class BaseConfig:
    config_version: int
    config_migrations: dict[int, Callable[[dict], dict]]

    def __init__(
            self,
            data: list[tuple[str, Any, type, bool]],
            config_version: int | None = None,
            config_migrations: dict[int, Callable[[dict], dict]] | None = None
    ):
        self.config_version = config_version if config_version is not None else 0
        self.config_migrations = config_migrations if config_migrations is not None else {}

        # The custom MageFlow branch needs saved model-specific experiment/
        # residency switches in the normal TrainConfig/UI state without bumping
        # OneTrainer's shared config schema. Add them only to TrainConfig
        # instances; old configs retain backward-compatible defaults.
        if self.__class__.__name__ == "TrainConfig":
            data = list(data)
            existing_names = {entry[0] for entry in data}
            if "self_flow_ema_on_gpu" not in existing_names:
                data.append(("self_flow_ema_on_gpu", False, bool, False))
            if "rlhf_dpo_linear_ema_on_gpu" not in existing_names:
                data.append(("rlhf_dpo_linear_ema_on_gpu", False, bool, False))
            if "rlhf_dpo_self_flow" not in existing_names:
                # Preserve the existing full chosen+rejected Self-Flow DPO
                # behavior unless the user explicitly opts out.
                data.append(("rlhf_dpo_self_flow", True, bool, False))
            if "rlhf_dpo_max_timestep" not in existing_names:
                # Literal fraction of the scheduler range. For 1000 training
                # timesteps, 0.95 means DPO samples are limited to t <= 950.
                data.append(("rlhf_dpo_max_timestep", 0.95, float, False))

        self.types = {}
        self.nullables = {}
        self.default_values = {}
        for (name, value, var_type, nullable) in data:
            setattr(self, name, value)
            self.types[name] = var_type
            self.nullables[name] = nullable
            self.default_values[name] = value

    def to_dict(self) -> dict:
        data = {
            '__version': self.config_version,
        }

        for name in self.types:
            value = getattr(self, name)
            if issubclass_safe(self.types[name], BaseConfig):
                data[name] = value.to_dict()
            elif self.types[name] is list or get_origin(self.types[name]) is list:
                if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[0], BaseConfig):
                    data[name] = [le.to_dict() for le in value] if value is not None else None
                else:
                    data[name] = value
            elif self.types[name] is dict or get_origin(self.types[name]) is dict:
                if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[1], BaseConfig):
                    dict_data = {}
                    for dict_key, dict_value in value.items():
                        dict_data[dict_key] = dict_value.to_dict()
                    data[name] = dict_data
                else:
                    data[name] = value
            elif self.types[name] is str:
                data[name] = value
            elif issubclass_safe(self.types[name], Enum):
                data[name] = None if value is None else str(value)
            elif self.types[name] is bool or self.types[name] is int:
                data[name] = value
            elif self.types[name] is float:
                if value in [float('inf'), float('-inf')]:
                    data[name] = str(value)
                else:
                    data[name] = value

        return data

    def from_dict(self, data: dict, migrate: bool = True) -> 'BaseConfig':
        if migrate:
            version = 0
            if '__version' in data:
                version = data['__version']

            while version in self.config_migrations:
                data = self.config_migrations[version](data)
                version += 1

        for name in self.types:
            try:
                if issubclass_safe(self.types[name], BaseConfig):
                    getattr(self, name).from_dict(data[name], migrate=migrate)
                elif self.types[name] is list or get_origin(self.types[name]) is list:
                    if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[0], BaseConfig):
                        list_type = get_args(self.types[name])[0]
                        if data[name] is not None:
                            old_value = \
                                getattr(self, name) if hasattr(self, name) and getattr(self, name) is not None else []
                            value = []
                            for i in range(len(data[name])):
                                if i < len(old_value) and i < len(data[name]):
                                    value.append(old_value[i].from_dict(data[name][i], migrate=migrate))
                                else:
                                    value.append(list_type.default_values().from_dict(data[name][i], migrate=migrate))
                        else:
                            value = None
                        setattr(self, name, value)
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] is dict or get_origin(self.types[name]) is dict:
                    if len(get_args(self.types[name])) > 0 and issubclass_safe(get_args(self.types[name])[1], BaseConfig):
                        dict_type = get_args(self.types[name])[1]
                        value = {}
                        for dict_key, dict_value in data[name].items():
                            value[dict_key] = dict_type.default_values().from_dict(dict_value, migrate=migrate)
                        setattr(self, name, value)
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] is str:
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else str(data[name]))
                    else:
                        setattr(self, name, str(data[name]))
                elif issubclass_safe(self.types[name], Enum):
                    if isinstance(data[name], str):
                        if self.nullables[name]:
                            setattr(self, name, None if data[name] is None else self.types[name][data[name]])
                        else:
                            setattr(self, name, self.types[name][data[name]])
                    else:
                        setattr(self, name, data[name])
                elif self.types[name] is bool:
                    setattr(self, name, data[name])
                elif self.types[name] is int:
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else int(data[name]))
                    else:
                        setattr(self, name, int(data[name]))
                elif self.types[name] is float:
                    # check for strings to support dicts loaded from json
                    if data[name] in [float('inf'), float('-inf'), 'inf', '-inf']:
                        setattr(self, name, float(data[name]))
                    if self.nullables[name]:
                        setattr(self, name, None if data[name] is None else float(data[name]))
                    else:
                        setattr(self, name, float(data[name]))
            except Exception:  # noqa: PERF203
                if name in data:
                    print(f"Could not set {name} as {str(data[name])}")
                else:
                    # print(f"Could not set {name}, not found.")
                    pass

        return self