from abc import ABC, abstractmethod
from numbers import Number
from typing import Dict, Generator
import numpy as np


class InvalidSchema:
    def __init__(self, message, schema):
        self.message = message
        self.schema = schema


class InvalidSchemaError(Exception):
    def __init__(self, invalid_schema: InvalidSchema):
        super().__init__(invalid_schema.message)
        self.invalid_schema = invalid_schema


class _InvalidSchemaConstructedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def to_schema_error(self, schema):
        return InvalidSchemaError(InvalidSchema(self.message, schema))


class SchemaMismatch:
    def __init__(self, message, schema, data):
        self.message = message
        self.schema = schema
        self.data = data


class SchemaMismatchError(Exception):
    def __init__(self, schema_mismatch: SchemaMismatch):
        super().__init__(schema_mismatch.message)
        self.schema_mismatch = schema_mismatch


class _StatNone:
    def __init__(self):
        pass


_stat_none = _StatNone()


class StatsType(ABC):
    @abstractmethod
    def generate(self, n_epochs: int):
        pass

    @abstractmethod
    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        pass

    @abstractmethod
    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        pass

    def valid(self, n_epochs: int, stats) -> bool:
        for _ in self.validate(n_epochs, stats):
            return False
        return True

    def valid_epoch(self, stats) -> bool:
        for _ in self.validate_epoch(stats):
            return False
        return True

    def assert_valid(self, n_epochs: int, stats) -> None:
        for mismatch in self.validate(n_epochs, stats):
            raise SchemaMismatchError(mismatch)

    def assert_valid_epoch(self, stats) -> None:
        for mismatch in self.validate_epoch(stats):
            raise SchemaMismatchError(mismatch)


class PreStatsType(ABC):
    @abstractmethod
    def generate(self) -> StatsType:
        pass


class _PreStatsPrimitive(PreStatsType):
    def __init__(self, t: StatsType):
        self.t = t

    def generate(self) -> StatsType:
        return self.t


class _StatsEpochArray(StatsType):
    def __init__(self, dtype=np.float64):
        self.dtype = dtype

    def generate(self, n_epochs: int):
        return np.zeros(n_epochs, dtype=self.dtype)

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, np.ndarray):
            yield SchemaMismatch(
                f"Epoch array expected a numpy array, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        if stats.shape != (n_epochs,):
            yield SchemaMismatch(
                f"Epoch array expected numpy array of shape {(n_epochs,)}, received {stats.shape}",
                self,
                stats,
            )
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, Number):
            yield SchemaMismatch(
                f"Epoch array expected a number, received {type(stats).__name__}",
                self,
                stats,
            )
        return


class StatsEpochArray(_PreStatsPrimitive):
    def __init__(self, dtype=np.float64):
        super().__init__(_StatsEpochArray(dtype))


class _StatsOpt(StatsType):
    def __init__(self, t: StatsType):
        if not isinstance(t, StatsType):
            raise _InvalidSchemaConstructedError(
                f"Stats opt expects only type args, received {t}"
            )
        self.t = t

    def generate(self, n_epochs: int):
        return _stat_none

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, _StatNone):
            for mismatch in self.t.validate(n_epochs, stats):
                yield mismatch
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, _StatNone):
            for mismatch in self.t.validate_epoch(stats):
                yield mismatch
        return


class StatsOpt(PreStatsType):
    def __init__(self, t):
        self.t = t

    def generate(self) -> StatsType:
        return _StatsOpt(generate_stats_schema(self.t))


class _StatsStruct(StatsType):
    def __init__(self, schema: Dict[str, StatsType], strict: bool = True):
        for key in schema:
            if not isinstance(key, str):
                raise _InvalidSchemaConstructedError(
                    f"Stats struct expects only string keys, received {key}"
                )
            if not isinstance(schema[key], StatsType):
                raise _InvalidSchemaConstructedError(
                    f"Stats struct expects only type values, received {schema[key]}"
                )
        self.schema = schema
        self.strict = strict

    def generate(self, n_epochs: int):
        return {key: self.schema[key].generate(n_epochs) for key in self.schema}

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, dict):
            yield SchemaMismatch(
                f"Stats struct expected a dictionary, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        for key in stats:
            if not isinstance(key, str):
                yield SchemaMismatch(
                    f"Stats struct expects only string keys, received {key}",
                    self,
                    stats,
                )
        for key in self.schema:
            stats_elem = stats.get(key, _stat_none)
            for mismatch in self.schema[key].validate(n_epochs, stats_elem):
                yield mismatch
        if self.strict:
            for key in stats:
                if key not in self.schema:
                    yield SchemaMismatch(
                        f"Stats struct received unspecified key {key}", self, stats
                    )
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, dict):
            yield SchemaMismatch(
                f"Stats struct expected a dictionary, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        for key in stats:
            if not isinstance(key, str):
                yield SchemaMismatch(
                    f"Stats struct expects only string keys, received {key}",
                    self,
                    stats,
                )
        for key in self.schema:
            stats_elem = stats.get(key, _stat_none)
            for mismatch in self.schema[key].validate_epoch(stats_elem):
                yield mismatch
        if self.strict:
            for key in stats:
                if key not in self.schema:
                    yield SchemaMismatch(
                        f"Stats struct received unspecified key {key}", self, stats
                    )
        return


class StatsStruct(PreStatsType):
    def __init__(self, schema: dict, strict: bool = True):
        if not isinstance(schema, dict):
            raise _InvalidSchemaConstructedError(
                f"Stats struct expects a dictionary schema, received {type(schema).__name__}"
            )
        self.schema = schema
        self.strict = strict

    def generate(self) -> StatsType:
        return _StatsStruct(
            {key: generate_stats_schema(self.schema[key]) for key in self.schema},
            strict=self.strict,
        )


class _StatsDict(StatsType):
    def __init__(self, t: StatsType):
        if not isinstance(t, StatsType):
            raise _InvalidSchemaConstructedError(
                f"Stats dict expects only type args, received {t}"
            )
        self.t = t

    def generate(self, n_epochs: int):
        return {}

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, dict):
            yield SchemaMismatch(
                f"Stats dict expected a dictionary, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        for key in stats:
            if not isinstance(key, str):
                yield SchemaMismatch(
                    f"Stats dict expects only string keys, received {key}",
                    self,
                    stats,
                )
            for mismatch in self.t.validate(n_epochs, stats[key]):
                yield mismatch
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, dict):
            yield SchemaMismatch(
                f"Stats dict expected a dictionary, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        for key in stats:
            if not isinstance(key, str):
                yield SchemaMismatch(
                    f"Stats dict expects only string keys, received {key}",
                    self,
                    stats,
                )
            for mismatch in self.t.validate_epoch(stats[key]):
                yield mismatch
        return


class StatsDict(PreStatsType):
    def __init__(self, t):
        self.t = t

    def generate(self) -> StatsType:
        return _StatsDict(generate_stats_schema(self.t))


class _StatsTrainVal(StatsType):
    def __init__(self, t: StatsType):
        if not isinstance(t, StatsType):
            raise _InvalidSchemaConstructedError(
                f"Stats train/val expects only type args, received {t}"
            )
        self.t = t

    def generate(self, n_epochs: int):
        return {"train": self.t.generate(n_epochs), "val": self.t.generate(n_epochs)}

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, dict):
            yield SchemaMismatch(
                f"Stats train/val expected a dictionary, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        if "train" not in stats:
            yield SchemaMismatch("Stats train/val expected 'train' key", self, stats)
        else:
            for mismatch in self.t.validate(n_epochs, stats["train"]):
                yield mismatch
        if "val" not in stats:
            yield SchemaMismatch("Stats train/val expected 'val' key", self, stats)
        else:
            for mismatch in self.t.validate(n_epochs, stats["val"]):
                yield mismatch
        for key in stats:
            if key not in ("train", "val"):
                yield SchemaMismatch(
                    f"Stats train/val received unexpected key {key}", self, stats
                )
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, dict):
            yield SchemaMismatch(
                f"Stats train/val expected a dictionary, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        if "train" not in stats:
            yield SchemaMismatch("Stats train/val expected 'train' key", self, stats)
        else:
            for mismatch in self.t.validate_epoch(stats["train"]):
                yield mismatch
        if "val" not in stats:
            yield SchemaMismatch("Stats train/val expected 'val' key", self, stats)
        else:
            for mismatch in self.t.validate_epoch(stats["val"]):
                yield mismatch
        for key in stats:
            if key not in ("train", "val"):
                yield SchemaMismatch(
                    f"Stats train/val received unexpected key {key}", self, stats
                )
        return


class StatsTrainVal(PreStatsType):
    def __init__(self, t):
        self.t = t

    def generate(self) -> StatsType:
        return _StatsTrainVal(generate_stats_schema(self.t))


class _StatsEpochMatrix(StatsType):
    def __init__(self, n_features: int, dtype=np.float64):
        if not isinstance(n_features, int):
            raise _InvalidSchemaConstructedError(
                f"Stats epoch matrix expects integer n_features, received {n_features}"
            )
        if n_features < 1:
            raise _InvalidSchemaConstructedError(
                f"Stats epoch matrix expects n_features >= 1, received {n_features}"
            )
        self.n_features = n_features
        self.dtype = dtype

    def generate(self, n_epochs: int):
        return np.zeros((n_epochs, self.n_features), dtype=self.dtype)

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, np.ndarray):
            yield SchemaMismatch(
                f"Epoch matrix expected a numpy array, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        expected_shape = (n_epochs, self.n_features)
        if stats.shape != expected_shape:
            yield SchemaMismatch(
                f"Epoch matrix expected shape {expected_shape}, received {stats.shape}",
                self,
                stats,
            )
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, np.ndarray):
            yield SchemaMismatch(
                f"Epoch matrix expected a numpy array for epoch, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        if stats.shape != (self.n_features,):
            yield SchemaMismatch(
                f"Epoch matrix expected epoch shape {(self.n_features,)}, received {stats.shape}",
                self,
                stats,
            )
        return


class StatsEpochMatrix(PreStatsType):
    def __init__(self, n_features: int, dtype=np.float64):
        self.n_features = n_features
        self.dtype = dtype

    def generate(self) -> StatsType:
        return _StatsEpochMatrix(self.n_features, self.dtype)


class _StatsAccumulator(StatsType):
    def __init__(self, reduce: str = "mean"):
        valid_reduce = ("mean", "sum", "min", "max", "last")
        if reduce not in valid_reduce:
            raise _InvalidSchemaConstructedError(
                f"Stats accumulator expects reduce in {valid_reduce}, received {reduce}"
            )
        self.reduce = reduce

    def generate(self, n_epochs: int):
        return np.zeros(n_epochs, dtype=np.float64)

    def validate(self, n_epochs: int, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, np.ndarray):
            yield SchemaMismatch(
                f"Stats accumulator expected a numpy array, received {type(stats).__name__}",
                self,
                stats,
            )
            return
        if stats.shape != (n_epochs,):
            yield SchemaMismatch(
                f"Stats accumulator expected shape {(n_epochs,)}, received {stats.shape}",
                self,
                stats,
            )
        return

    def validate_epoch(self, stats) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(stats, Number):
            yield SchemaMismatch(
                f"Stats accumulator expected a number, received {type(stats).__name__}",
                self,
                stats,
            )
        return


class StatsAccumulator(_PreStatsPrimitive):
    def __init__(self, reduce: str = "mean"):
        super().__init__(_StatsAccumulator(reduce))


def generate_stats_schema(schema) -> StatsType:
    try:
        match schema:
            case dict():
                return _StatsStruct(
                    {key: generate_stats_schema(schema[key]) for key in schema}
                )
            case PreStatsType():
                return schema.generate()
            case StatsType():
                return schema
            case _:
                if schema is float:
                    return _StatsEpochArray(np.float64)
                if schema is int:
                    return _StatsEpochArray(np.int64)
                if schema == "epoch_array":
                    return _StatsEpochArray()
                if schema == "train_val":
                    return _StatsTrainVal(_StatsEpochArray())
                raise InvalidSchemaError(
                    InvalidSchema(f"Unrecognised stats schema object {schema}", schema)
                )
    except _InvalidSchemaConstructedError as e:
        raise e.to_schema_error(schema)
