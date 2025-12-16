from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import fields
from types import TracebackType
from typing import Self, final, override

import optuna
from pydantic.dataclasses import dataclass


@dataclass
class Params(ABC):
    @classmethod
    @abstractmethod
    def define(cls, space: ParameterSpace) -> None: ...

    @classmethod
    def init_from(
        cls,
        source: ParameterSpace
        | optuna.trial.Trial
        | optuna.trial.FrozenTrial
        | dict[str, object],
    ) -> Self:
        if not isinstance(source, ParameterSpace):
            space = ParameterSpace(source)
        else:
            space = source
        configurator = space.configure(cls)
        with configurator as space:
            cls.define(space)
        return configurator.instance

    def to_dict(self) -> dict[str, object]:
        return {
            field.name: (
                value.to_dict()
                if isinstance(value := getattr(self, field.name), Params)
                else value
            )
            for field in fields(self)
        }


@final
class Condition:
    """
    Result of a comparison (e.g., param == 'value').
    """

    def __init__(
        self,
        is_match: bool,
        parent_space: ParameterSpace,
        name: str,
        value_suffix: str,
    ):
        self._is_match = is_match
        self._parent_space = parent_space
        self._name = name
        self._value_suffix = str(value_suffix)

    def __bool__(self) -> bool:
        return self._is_match

    def __enter__(self) -> ParameterSpace:
        active = self._parent_space.is_active and self._is_match
        child_name = f"{self._name}{self._parent_space.delimiter}{self._value_suffix}"
        return self._parent_space.register_subspace(child_name, active=active)

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


class Suggestion:
    def __init__(self, value: object, space: ParameterSpace, name: str):
        self._value: object = value
        self._space: ParameterSpace = space
        self._name: str = name

    @property
    def value(self) -> object:
        return self._value

    def matches(self, other: object) -> Condition:
        is_match = (self.value == other) if self.value is not None else False
        return Condition(is_match, self._space, self._name, value_suffix=str(other))

    def choice(self) -> Condition:
        return Condition(True, self._space, self._name, value_suffix=str(self.value))

    @override
    def __eq__(self, other: object) -> bool:
        return self.value == other

    @override
    def __ne__(self, other: object) -> bool:
        return self.value != other

    @override
    def __str__(self) -> str:
        return str(self.value)

    def __int__(self) -> int:
        if not isinstance(self.value, int):
            raise TypeError(f"Cannot convert {self.value} to int")
        return int(self.value)

    def __float__(self) -> float:
        if not isinstance(self.value, float):
            raise TypeError(f"Cannot convert {self.value} to float")
        return float(self.value)

    def __bool__(self) -> bool:
        return bool(self.value)


@final
class BooleanSuggestion(Suggestion):
    def __enter__(self) -> ParameterSpace:
        # To allow 'with boolean_suggestion' as a shorthand for 'with boolean_suggestion.matches(True)'
        return self.matches(True).__enter__()

    def __exit__(
        self,
        type_: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        return None


@final
class Configurator[T]:
    """
    Context manager that accumulates parameters in a buffer and
    instantiates the target class on exit.
    """

    def __init__(self, space: ParameterSpace, output_type: type[T]):
        self._space = space
        self._output_type = output_type
        self._buffer: dict[str, object] = {}
        self._instance: T | None = None

    def __enter__(self) -> ParameterSpace:
        # Create a special space that writes to this configurator's buffer
        return self._space.clone_with_buffer(self._buffer)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type:
            return
        try:
            # Instantiate the object using the collected parameters
            self._instance = self._output_type(**self._buffer)
        except TypeError as e:
            raise TypeError(f"Failed to instantiate {self._output_type.__name__}: {e}")

    @property
    def instance(self) -> T:
        if self._instance is None:
            raise ValueError(
                "Builder could not instantiate the object, you should enter its space via 'with configurator as space: ...' to define all required parameters in the space"
            )
        return self._instance


@final
class ParameterSpace:
    def __init__(
        self,
        source: optuna.Trial | optuna.trial.FrozenTrial | dict[str, object],
        prefix: str = "",
        delimiter: str = "-",
        active: bool = True,
        _params_buffer: dict[str, object] | None = None,
    ):
        self._source = source
        self._prefix = prefix
        self._delimiter = delimiter
        self._active = active
        self._params_buffer = _params_buffer

        self._sampled_params: dict[str, object] | None = None
        if isinstance(source, dict):
            self._sampled_params = source
        elif isinstance(source, optuna.trial.FrozenTrial):
            self._sampled_params = source.params
        # else: source is optuna Trial (during an optuna Study), _sampled_params remains None

    @property
    def is_active(self) -> bool:
        return self._active

    @property
    def delimiter(self) -> str:
        return self._delimiter

    def configure[T](self, output_type: type[T]) -> Configurator[T]:
        """
        Begins a definition block for a class.
        Returns a Configurator. Access .instance on the configurator after the block.
        """
        return Configurator(self, output_type)

    def assign(self, name: str, value: object) -> None:
        """
        Manually assign a value/object to the parameter set currently being defined.
        """
        if self._params_buffer is not None:
            self._params_buffer[name] = value

    def subspace(self, name: str, active: bool = True) -> ParameterSpace:
        is_active = self.is_active and active
        return self.register_subspace(name, active=is_active)

    def register_subspace(self, name: str, active: bool) -> ParameterSpace:
        full_prefix = f"{self._prefix}{self._delimiter}{name}" if self._prefix else name
        return ParameterSpace(
            self._source,
            full_prefix,
            self._delimiter,
            active=active,
            _params_buffer=None,
        )

    def clone_with_buffer(self, buffer: dict[str, object]) -> ParameterSpace:
        return ParameterSpace(
            self._source,
            self._prefix,
            self._delimiter,
            self._active,
            _params_buffer=buffer,
        )

    def _get_name(self, name: str) -> str:
        return f"{self._prefix}{self._delimiter}{name}" if self._prefix else name

    def _suggest_generic(
        self, method: str, name: str, *args: object, **kwargs: object
    ) -> Suggestion:
        full_name = self._get_name(name)

        if not self._active:
            return Suggestion(None, self, name)

        val = None

        if self._sampled_params is not None:
            # parsing from params
            if full_name in self._sampled_params:
                val = self._sampled_params[full_name]
            else:
                raise KeyError(f"Missing parameter in source: {full_name}")
        else:
            assert isinstance(self._source, optuna.trial.Trial)
            func = getattr(self._source, method)
            val = func(full_name, *args, **kwargs)

        if self._params_buffer is not None:
            self._params_buffer[name] = val

        return Suggestion(val, self, name)

    def suggest_categorical(self, name: str, choices: Sequence[object]) -> Suggestion:
        return self._suggest_generic("suggest_categorical", name, choices)

    def suggest_int(self, name: str, low: int, high: int, step: int = 1) -> Suggestion:
        return self._suggest_generic("suggest_int", name, low, high, step=step)

    def suggest_float(
        self, name: str, low: float, high: float, step: float | None = None
    ) -> Suggestion:
        return self._suggest_generic("suggest_float", name, low, high, step=step)

    def suggest_bool(self, name: str) -> BooleanSuggestion:
        full_name = self._get_name(name)

        if not self._active:
            return BooleanSuggestion(None, self, name)

        val = None

        if self._sampled_params is not None:
            # parsing from params
            if full_name in self._sampled_params:
                val = self._sampled_params[full_name]
            else:
                raise KeyError(f"Missing parameter in source: {full_name}")
        else:
            assert isinstance(self._source, optuna.trial.Trial)
            val = self._source.suggest_categorical(full_name, [True, False])

        if self._params_buffer is not None and val is not None:
            self._params_buffer[name] = val

        return BooleanSuggestion(val, self, name)
