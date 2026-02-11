from abc import ABC, abstractmethod
from typing import Callable, Dict, Generator, Iterable, List, Optional


class InvalidSchema:
    def __init__(self, message, schema):
        self.message = message
        self.schema = schema


class InvalidSchemaError(Exception):
    def __init__(self, invalid_schema: InvalidSchema):
        super().__init__(invalid_schema.message)
        self.invalid_schema = invalid_schema


def invalid_schema_error(message, schema):
    return InvalidSchemaError(InvalidSchema(message, schema))


class _InvalidSchemaConstructedError(Exception):
    def __init__(self, message: str):
        super().__init__(message)

    def to_schema_error(self, schema):
        return invalid_schema_error(self.message, schema)


class SchemaMismatch:
    def __init__(self, message, schema, data):
        self.message = message
        self.schema = schema
        self.data = data


class SchemaMismatchError(Exception):
    def __init__(self, schema_mismatch: SchemaMismatch):
        super().__init__(schema_mismatch.message)
        self.schema_mismatch = schema_mismatch


class _JSONNone:
    def __init__(self):
        pass


_json_none = _JSONNone()


class JSONType(ABC):
    @abstractmethod
    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        pass

    def valid(self, data) -> bool:
        for _ in self.validate(data):
            return False
        return True

    def assert_valid(self, data) -> None:
        for mismatch in self.validate(data):
            raise SchemaMismatchError(mismatch)


class PreJSONType(ABC):
    @abstractmethod
    def generate(self) -> JSONType:
        pass


class _JSONPrimitive(JSONType):
    def __init__(self, t, name):
        self.t = t
        self.name = name

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(data, self.t):
            yield SchemaMismatch(
                f"Schema expected {self.name}, received {data}", self, data
            )


class _PreJSONPrimitive(PreJSONType):
    def __init__(self, t: JSONType):
        self.t = t

    def generate(self) -> JSONType:
        return self.t


class _JSONBool(_JSONPrimitive):
    def __init__(self):
        super().__init__(bool, "bool")


_json_bool = _JSONBool()


class JSONBool(_PreJSONPrimitive):
    def __init__(self):
        super().__init__(_json_bool)


class _JSONInt(_JSONPrimitive):
    def __init__(self):
        super().__init__(int, "int")


_json_int = _JSONInt()


class JSONInt(_PreJSONPrimitive):
    def __init__(self):
        super().__init__(_json_int)


class _JSONFloat(_JSONPrimitive):
    def __init__(self):
        super().__init__(float, "float")


_json_float = _JSONFloat()


class JSONFloat(_PreJSONPrimitive):
    def __init__(self):
        super().__init__(_json_float)


class _JSONStr(_JSONPrimitive):
    def __init__(self):
        super().__init__(str, "str")


_json_str = _JSONStr()


class JSONStr(_PreJSONPrimitive):
    def __init__(self):
        super().__init__(_json_str)


class _JSONPredStr(JSONType):
    def __init__(self, preds: List[Callable[[str], Optional[str]]]):
        self.preds = preds

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(data, str):
            yield SchemaMismatch(f"Schema expected str, received {data}", self, data)
            return
        for pred in self.preds:
            result = pred(data)
            if result is not None:
                yield SchemaMismatch(result, self, data)


class JSONPredStr(_PreJSONPrimitive):
    def __init__(self, preds: List[Callable[[str], Optional[str]]]):
        super().__init__(_JSONPredStr(preds))


class _JSONAny(JSONType):
    def __init__(self):
        pass

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        match data:
            case bool():
                return
            case int():
                return
            case float():
                return
            case str():
                return
            case list():
                for elem in data:
                    yield from self.validate(elem)
            case dict():
                for key in data:
                    if not isinstance(key, str):
                        yield SchemaMismatch(
                            f"Schema expected only string keys, received {key}",
                            self,
                            data,
                        )
                    yield from self.validate(data[key])
            case _:
                yield SchemaMismatch(
                    f"Schema expected only valid JSON values, received {data}",
                    self,
                    data,
                )


_json_any = _JSONAny()


class JSONAny(_PreJSONPrimitive):
    def __init__(self):
        super().__init__(_json_any)


class _JSONOpt(JSONType):
    def __init__(self, t: JSONType):
        if not isinstance(t, JSONType):
            raise _InvalidSchemaConstructedError(
                f"Schema opt expected only type args, received {t}"
            )
        self.t = t

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(data, _JSONNone):
            yield from self.t.validate(data)


class JSONOpt(PreJSONType):
    def __init__(self, t):
        self.t = t

    def generate(self) -> JSONType:
        return _JSONOpt(generate_json_schema(self.t))


class _JSONLit(JSONType):
    def __init__(self, values: Iterable[JSONType]):
        try:
            iter(values)
        except TypeError:
            raise _InvalidSchemaConstructedError(
                f"Schema union expected iterable values, received {values}"
            )
        for value in values:
            if not _json_any.valid(value):
                raise _InvalidSchemaConstructedError(
                    f"Schema literal expected valid JSON literal, received {value}"
                )
        self.values = values

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if data not in self.values:
            yield SchemaMismatch(
                f"Schema expected a literal from {self.values}, received {data}",
                self,
                data,
            )


class JSONLit(_PreJSONPrimitive):
    def __init__(self, *args):
        super().__init__(_JSONLit(args))


class _JSONLenBoundArray(JSONType):
    def __init__(
        self, t: JSONType, min_len: Optional[int] = None, max_len: Optional[int] = None
    ):
        if not isinstance(t, JSONType):
            raise _InvalidSchemaConstructedError(
                f"Schema array expected only type args, received {t}"
            )
        if min_len is not None:
            if not isinstance(min_len, int):
                raise _InvalidSchemaConstructedError(
                    f"Schema array expected integer min length, received {min_len}"
                )
            if min_len < 0:
                raise _InvalidSchemaConstructedError(
                    f"Schema array expected min length >= 0, received {min_len}"
                )
        if max_len is not None:
            if not isinstance(max_len, int):
                raise _InvalidSchemaConstructedError(
                    f"Schema array expected integer max length, received {max_len}"
                )
            if max_len < 0:
                raise _InvalidSchemaConstructedError(
                    f"Schema array expected max length >= 0, received {max_len}"
                )
        if min_len is not None and max_len is not None and min_len > max_len:
            raise _InvalidSchemaConstructedError(
                f"Schema array expected max length >= min length, received {min_len} and {max_len}"
            )
        self.t = t
        self.min_len = min_len
        self.max_len = max_len

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(data, list):
            yield SchemaMismatch(
                f"Schema expected an array, received {data}", self, data
            )
            return
        if self.min_len is not None and self.min_len > len(data):
            yield SchemaMismatch(
                f"Schema expected array of at least {self.min_len} items, received {len(data)} items",
                self,
                data,
            )
        if self.max_len is not None and self.max_len < len(data):
            yield SchemaMismatch(
                f"Schema expected array of at most {self.max_len} items, received {len(data)} items",
                self,
                data,
            )
        for elem in data:
            yield from self.t.validate(elem)
        return


class JSONArray(PreJSONType):
    def __init__(self, t, min_len: Optional[int] = None, max_len: Optional[int] = None):
        self.t = t
        self.min_len = min_len
        self.max_len = max_len

    def generate(self) -> JSONType:
        return _JSONLenBoundArray(
            generate_json_schema(self.t), min_len=self.min_len, max_len=self.max_len
        )


class JSONLenBoundArray(PreJSONType):
    def __init__(self, t):
        self.t = t

    def generate(self) -> JSONType:
        return _JSONLenBoundArray(generate_json_schema(self.t))


class _JSONStruct(JSONType):
    def __init__(self, schema: Dict[str, JSONType], strict=True):
        for key in schema:
            if not isinstance(key, str):
                raise _InvalidSchemaConstructedError(
                    f"Schema struct expected only string keys, received {key}"
                )
            if not isinstance(schema[key], JSONType):
                raise _InvalidSchemaConstructedError(
                    f"Schema struct expected only type values, received {schema[key]}"
                )
        self.schema = schema
        self.strict = strict

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(data, dict):
            yield SchemaMismatch(
                f"Schema expected a dictionary, received {data}", self, data
            )
            return
        for key in data:
            if not isinstance(key, str):
                yield SchemaMismatch(
                    f"Schema expected only string keys, received {key}", self, data
                )
        for key in self.schema:
            data_elem = data.get(key, _json_none)
            yield from self.schema[key].validate(data_elem)
        if self.strict:
            for key in data:
                if key not in self.schema:
                    yield SchemaMismatch(
                        f"Schema expected no extra keys, received {key}", self, data
                    )
        return


class JSONStruct(PreJSONType):
    def __init__(self, schema):
        self.schema = schema

    def generate(self) -> JSONType:
        return _JSONStruct(
            {key: generate_json_schema(self.schema[key]) for key in self.schema}
        )


class _JSONDict(JSONType):
    def __init__(self, t: JSONType):
        if not isinstance(t, JSONType):
            raise _InvalidSchemaConstructedError(
                f"Schema dict expected only type args, received {t}"
            )
        self.t = t

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        if not isinstance(data, dict):
            yield SchemaMismatch(
                f"Schema expected a dictionary, received {data}", self, data
            )
            return
        for key in data:
            if not isinstance(key, str):
                yield SchemaMismatch(
                    f"Schema expected only string keys, received {key}", self, data
                )
            yield from self.t.validate(data[key])
        return


class JSONDict(PreJSONType):
    def __init__(self, t):
        self.t = t

    def generate(self) -> JSONType:
        return _JSONDict(generate_json_schema(self.t))


class _JSONUnion(JSONType):
    def __init__(self, ts: Iterable[JSONType]):
        try:
            iter(ts)
        except TypeError:
            raise _InvalidSchemaConstructedError(
                f"Schema union expected iterable type args, received {t}"
            )
        for t in ts:
            if not isinstance(t, JSONType):
                raise _InvalidSchemaConstructedError(
                    f"Schema union expected only type args, received {t}"
                )
        self.ts = ts

    def validate(self, data) -> Generator[SchemaMismatch, None, None]:
        for t in self.ts:
            if t.valid(data):
                return
        yield SchemaMismatch(
            "Schema expected a match in union, received none", self, data
        )


class JSONUnion(PreJSONType):
    def __init__(self, ts):
        self.ts = ts

    def generate(self) -> JSONType:
        return _JSONUnion([generate_json_schema(t) for t in self.ts])


def _generate_json_array_schema(schema):
    match len(schema):
        case 1:
            return _JSONLenBoundArray(generate_json_schema(schema[0]))
        case 2:
            match schema[1]:
                case int():
                    return _JSONLenBoundArray(
                        generate_json_schema(schema[0]),
                        min_len=schema[1],
                        max_len=schema[1],
                    )
                case list():
                    match len(schema[1]):
                        case 0:
                            return _JSONLenBoundArray(generate_json_schema(schema[0]))
                        case 1:
                            return _JSONLenBoundArray(
                                generate_json_schema(schema[0]), max_len=schema[1][0]
                            )
                        case 2:
                            return _JSONLenBoundArray(
                                generate_json_schema(schema[0]),
                                min_len=schema[1][0],
                                max_len=schema[1][1],
                            )
        case numargs:
            raise invalid_schema_error(
                f"Schema array expected 1-2 arguments, received {numargs}",
                schema,
            )


def _generate_json_schema_from_tuple(schema):
    if len(schema) == 0:
        raise invalid_schema_error(
            f"Schema tuple expected at least 1 argument, received 0",
            schema,
        )
    match schema[0]:
        case "bool":
            if len(schema) != 1:
                raise invalid_schema_error(
                    f"Schema bool tuple expected 1 argument, received {len(schema)}",
                    schema,
                )
            return _json_bool
        case "int":
            if len(schema) != 1:
                raise invalid_schema_error(
                    f"Schema int tuple expected 1 argument, received {len(schema)}",
                    schema,
                )
            return _json_int
        case "float":
            if len(schema) != 1:
                raise invalid_schema_error(
                    f"Schema float tuple expected 1 argument, received {len(schema)}",
                    schema,
                )
            return _json_bool
        case "str":
            if len(schema) != 1:
                raise invalid_schema_error(
                    f"Schema str tuple expected 1 argument, received {len(schema)}",
                    schema,
                )
            return _json_str
        case "any":
            if len(schema) != 1:
                raise invalid_schema_error(
                    f"Schema any tuple expected 1 argument, received {len(schema)}",
                    schema,
                )
            return _json_any
        case "dict":
            if len(schema) != 1:
                raise invalid_schema_error(
                    f"Schema dict tuple expected 1 argument, received {len(schema)}",
                    schema,
                )
            return _JSONDict(generate_json_schema(schema[1]))
        case "lit":
            return _JSONLit(schema[1:])
        case "opt":
            if len(schema) != 2:
                raise invalid_schema_error(
                    f"Schema opt tuple expected 2 arguments, received {len(schema)}",
                    schema,
                )
            return _JSONOpt(generate_json_schema(schema[1]))
        case "union":
            return _JSONUnion((generate_json_schema(t) for t in schema[1:]))


def generate_json_schema(schema):
    try:
        match schema:
            case dict():
                return _JSONStruct(
                    {key: generate_json_schema(schema[key]) for key in schema}
                )
            case list():
                return _generate_json_array_schema(schema)
            case tuple():
                return _generate_json_schema_from_tuple(schema)
            case PreJSONType():
                return schema.generate()
            case JSONType():
                return schema
            case _:
                if schema is bool or schema == "bool":
                    return _JSONBool()
                if schema is int or schema == "int":
                    return _JSONInt()
                if schema is float or schema == "float":
                    return _JSONFloat()
                if schema is str or schema == "str":
                    return _JSONStr()
                if schema == "any":
                    return _json_any
                raise invalid_schema_error(
                    f"Unrecognised schema object {schema}", schema
                )
    except _InvalidSchemaConstructedError as e:
        raise e.to_schema_error(schema)
