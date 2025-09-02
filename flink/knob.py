from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from utils.config import DefaultKnobConfig, KnobConfig
from utils.types import KnobUnit, StorageUnit, TimeUnit


class KnobType(Enum):
    INTEGER = 0
    FLOAT = 1
    BOOLEAN = 2
    STORAGE_SIZE = 3
    DURATION = 4
    ENUM = 5

    def is_integer(self):
        return self == KnobType.INTEGER

    def is_float(self):
        return self == KnobType.FLOAT

    def is_boolean(self):
        return self == KnobType.BOOLEAN

    def is_storage_size(self):
        return self == KnobType.STORAGE_SIZE

    def is_duration(self):
        return self == KnobType.DURATION

    def is_enum(self):
        return self == KnobType.ENUM

    def is_unit(self):
        return self in [KnobType.STORAGE_SIZE, KnobType.DURATION]


@dataclass
class KnobDef:
    name: str
    knob_type: KnobType
    default: Any
    value_range: Any
    power_of_two: Optional[bool] = False

    def to_dict(self):
        if self.knob_type.is_unit():
            return {
                "name": self.name,
                "type": self.knob_type.name.lower(),
                "default": str(self.default),
                "min": str(self.value_range[0]),
                "max": str(self.value_range[1]),
                "power_of_two": self.power_of_two,
            }
        elif self.knob_type.is_enum():
            return {
                "name": self.name,
                "type": self.knob_type.name.lower(),
                "default": self.default,
                "values": self.value_range,
            }
        elif self.knob_type.is_boolean():
            return {
                "name": self.name,
                "type": self.knob_type.name.lower(),
                "default": str(self.default).lower(),
            }
        elif self.knob_type.is_integer() or self.knob_type.is_float():
            return {
                "name": self.name,
                "type": self.knob_type.name.lower(),
                "default": self.default,
                "min": self.value_range[0],
                "max": self.value_range[1],
                "power_of_two": self.power_of_two,
            }
        else:
            raise ValueError(f"Unknown knob type: {self.knob_type.name}")

    def __init__(
        self,
        name: str,
        knob_type: KnobType,
        default: Any,
        value_range: Any = None,
        power_of_two: bool = False,
    ):
        """
        This class represents a tunable knob.

        :param name: knob name
        :param knob_type: knob type
        :param default: default value
        :param value_range: range. Should be [min, max] for integer and float, [val1, val2, ...] for categorical,
        not required for boolean
        """
        self.name = name
        self.knob_type = knob_type
        self.default = default
        self.value_range = value_range
        self.power_of_two = power_of_two == True
        try:
            self._sanity_check(knob_type, default, value_range, power_of_two)
        except AssertionError:
            raise ValueError(
                f"Invalid knob parameters: name={name}, knob_type={knob_type}, default={default}, value_range={value_range}, power_of_two={power_of_two}"
            )

    def get_percentile_from_value(self, value: Any) -> float:
        if self.knob_type.is_integer():
            min_val, max_val = self.value_range
            return (int(value) - min_val) / (max_val - min_val)
        elif self.knob_type.is_float():
            min_val, max_val = self.value_range
            return (float(value) - min_val) / (max_val - min_val)
        elif self.knob_type.is_enum():
            index = self.value_range.index(value)
            return index / (len(self.value_range) - 1)
        elif self.knob_type.is_boolean():
            return 0.0 if (not value or value.lower() == "false") else 1.0
        elif self.knob_type.is_storage_size():
            min_val, max_val = self.value_range
            return (StorageUnit.from_str(value).value() - min_val.value()) / (
                max_val.value() - min_val.value()
            )
        elif self.knob_type.is_duration():
            min_val, max_val = self.value_range
            return (TimeUnit.from_str(value).value() - min_val.value()) / (
                max_val.value() - min_val.value()
            )
        else:
            raise ValueError("Cannot convert value to percentile for boolean knob")

    def get_value_from_percentile(self, percentile: float) -> Any:
        assert 0 <= percentile <= 1, "Percentile should be between 0 and 1"
        if self.knob_type.is_integer():
            min_val, max_val = self.value_range
            return self._get_int_value_from_percentile(percentile, min_val, max_val)
        elif self.knob_type.is_float():
            min_val, max_val = self.value_range
            return float(percentile * (max_val - min_val) + min_val)
        elif self.knob_type.is_enum():
            index = int(percentile * len(self.value_range))
            index = max(0, min(index, len(self.value_range) - 1))
            return self.value_range[index]
        elif self.knob_type.is_boolean():
            if percentile < 0.5:
                return False
            else:
                return True
        elif self.knob_type.is_storage_size():
            min_val, max_val = self.value_range
            return StorageUnit(
                self._get_int_value_from_percentile(
                    percentile, min_val.value(), max_val.value()
                ),
                min_val.unit(),
            )
        elif self.knob_type.is_duration():
            min_val, max_val = self.value_range
            return TimeUnit(
                self._get_int_value_from_percentile(
                    percentile, min_val.value(), max_val.value()
                ),
                min_val.unit(),
            )
        else:
            raise ValueError("Cannot convert percentile to value for boolean knob")

    def _get_int_value_from_percentile(
        self, percentile: float, min_val: int, max_val: int
    ) -> int:
        if self.power_of_two:
            log_min_val = min_val.bit_length()
            log_max_val = max_val.bit_length()
            return 2 ** int(
                percentile * (log_max_val - log_min_val + 1) + log_min_val - 1
            )
        else:
            return int(percentile * (max_val - min_val + 1)) + min_val

    @staticmethod
    def _sanity_check(
        knob_type: KnobType,
        default_value: Any,
        value_range: Any,
        power_of_two: bool,
    ) -> None:
        if knob_type.is_integer():
            assert isinstance(default_value, int)
            assert isinstance(value_range, list)
            assert len(value_range) == 2
            assert isinstance(value_range[0], int)
            assert isinstance(value_range[1], int)
            assert value_range[0] <= default_value <= value_range[1]
            if power_of_two:
                assert default_value & (default_value - 1) == 0
                assert value_range[0] & (value_range[0] - 1) == 0
                assert value_range[1] & (value_range[1] - 1) == 0
        elif knob_type.is_float():
            assert isinstance(default_value, float)
            assert isinstance(value_range, list)
            assert len(value_range) == 2
            assert isinstance(value_range[0], float)
            assert isinstance(value_range[1], float)
            assert value_range[0] <= default_value <= value_range[1]
            assert not power_of_two
        elif knob_type.is_unit():
            assert isinstance(default_value, KnobUnit)
            assert isinstance(value_range, list)
            assert len(value_range) == 2
            assert isinstance(value_range[0], KnobUnit)
            assert isinstance(value_range[1], KnobUnit)
            assert value_range[0].unit() == default_value.unit()
            assert value_range[1].unit() == default_value.unit()
            assert (
                value_range[0].value()
                <= default_value.value()
                <= value_range[1].value()
            )
            if power_of_two:
                assert default_value.value() & (default_value.value() - 1) == 0
                assert value_range[0].value() & (value_range[0].value() - 1) == 0
                assert value_range[1].value() & (value_range[1].value() - 1) == 0
        elif knob_type.is_enum():
            assert isinstance(value_range, list)
            assert len(value_range) > 0
            assert default_value in value_range
            assert not power_of_two
        elif knob_type.is_boolean():
            assert isinstance(default_value, bool)
            assert (
                value_range is None
                or value_range == [True, False]
                or value_range == [False, True]
            )
            assert not power_of_two

    def format_value(self, value: Any):
        if self.knob_type.is_integer():
            return int(value)
        elif self.knob_type.is_float():
            return float(value)
        elif self.knob_type.is_unit():
            return str(value)
        elif self.knob_type.is_enum():
            return str(value)
        elif self.knob_type.is_boolean():
            return "true" if value else "false"
        elif self.knob_type.is_boolean():
            return value
        else:
            raise ValueError("Cannot convert value to string")

    def __repr__(self):
        return f"KnobDef(name={self.name}, type={self.knob_type}, default={self.default}, range={self.value_range}, power_of_two={self.power_of_two})"

    def __str__(self):
        return self.__repr__()


def parse_knob_def(
    knob_config: list[KnobConfig], excluded_prefixes: Optional[list[str]] = None
) -> list[KnobDef]:
    knobs = list[KnobDef]()
    excluded = tuple(excluded_prefixes) if excluded_prefixes else tuple()
    for knob in knob_config:
        name = knob.name

        if name.startswith(excluded):
            continue

        power_of_two = knob.power_of_two
        match knob.type:
            case "integer":
                knob_type = KnobType.INTEGER
                default_value = int(knob.default)
                value_range = [int(knob.min), int(knob.max)]
                knobs.append(
                    KnobDef(
                        name,
                        knob_type,
                        default_value,
                        value_range,
                        power_of_two,
                    )
                )
            case "float":
                knob_type = KnobType.FLOAT
                default_value = float(knob.default)
                value_range = [float(knob.min), float(knob.max)]
                knobs.append(
                    KnobDef(
                        name,
                        knob_type,
                        default_value,
                        value_range,
                        power_of_two,
                    )
                )
            case "duration":
                knob_type = KnobType.DURATION
                default_value = TimeUnit.from_str(knob.default)
                value_range = [
                    TimeUnit.from_str(knob.min),
                    TimeUnit.from_str(knob.max),
                ]
                knobs.append(
                    KnobDef(
                        name,
                        knob_type,
                        default_value,
                        value_range,
                        power_of_two,
                    )
                )
            case "storage_size":
                knob_type = KnobType.STORAGE_SIZE
                default_value = StorageUnit.from_str(knob.default)
                value_range = [
                    StorageUnit.from_str(knob.min),
                    StorageUnit.from_str(knob.max),
                ]
                knobs.append(
                    KnobDef(
                        name,
                        knob_type,
                        default_value,
                        value_range,
                        power_of_two,
                    )
                )
            case "enum":
                knob_type = KnobType.ENUM
                default_value = knob.default
                value_range = knob.values
                knobs.append(
                    KnobDef(
                        name,
                        knob_type,
                        default_value,
                        value_range,
                        power_of_two,
                    )
                )
            case "boolean":
                knob_type = KnobType.BOOLEAN
                default_value = bool(knob.default)
                value_range = None
                knobs.append(
                    KnobDef(
                        name,
                        knob_type,
                        default_value,
                        value_range,
                        power_of_two,
                    )
                )
            case _:
                raise ValueError(f"Unknown knob type: {knob.type}")

    return knobs


def parse_default_knob_def(
    default_knob_config: list[DefaultKnobConfig],
) -> dict[str, str]:
    default_knobs = dict[str, str]()
    for knob in default_knob_config:
        name = knob.name
        value = knob.value
        default_knobs[name] = value
    return default_knobs
