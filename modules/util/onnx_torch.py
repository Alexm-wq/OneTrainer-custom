"""Run small ONNX models as differentiable torch modules, without the onnx package.

Reads the protobuf wire format directly and supports the operators that exported
convolutional networks such as insightface's ArcFace models use.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

# TensorProto data types.
DTYPES = {1: np.float32, 6: np.int32, 7: np.int64, 10: np.float16, 11: np.float64}


def _varint(data, index):
    result = shift = 0
    while True:
        byte = data[index]
        index += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result - (1 << 64) if result >= 1 << 63 else result, index
        shift += 7


def _fields(data):
    index = 0
    while index < len(data):
        key, index = _varint(data, index)
        number, wire = key >> 3, key & 7
        if wire == 0:
            value, index = _varint(data, index)
        elif wire == 1:
            value, index = data[index:index + 8], index + 8
        elif wire == 2:
            length, index = _varint(data, index)
            value, index = data[index:index + length], index + length
        elif wire == 5:
            value, index = data[index:index + 4], index + 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
        yield number, wire, value


def _ints(wire, value):
    if wire == 0:
        return [value]
    values, index = [], 0
    while index < len(value):
        item, index = _varint(value, index)
        values.append(item)
    return values


def _tensor(data):
    dims, dtype, name, raw, values = [], 1, "", None, []
    for number, wire, value in _fields(data):
        if number == 1:
            dims += _ints(wire, value)
        elif number == 2:
            dtype = value
        elif number == 4:
            values.append(np.frombuffer(value, dtype="<f4"))
        elif number in (5, 7):
            values.append(np.array(_ints(wire, value)))
        elif number == 8:
            name = bytes(value).decode()
        elif number == 9:
            raw = value
        elif number == 10:
            values.append(np.frombuffer(value, dtype="<f8"))
        elif number == 13:
            raise ValueError(f"ONNX tensor {name!r} uses external data, which is not supported")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported ONNX tensor type {dtype} for {name!r}")
    if raw is not None:
        array = np.frombuffer(raw, dtype=np.dtype(DTYPES[dtype]).newbyteorder("<"))
    else:
        array = np.concatenate(values) if values else np.zeros(0)
    array = array.astype(DTYPES[dtype]).reshape(dims)
    return name, torch.from_numpy(array.copy())


def _attribute(data):
    name, kind, fields = "", 0, {}
    for number, wire, value in _fields(data):
        if number == 1:
            name = bytes(value).decode()
        elif number == 20:
            kind = value
        elif number in (7, 8):
            if number == 7:
                items = list(np.frombuffer(value, dtype="<f4")) if wire == 2 else [np.frombuffer(value, dtype="<f4")[0]]
            else:
                items = _ints(wire, value)
            fields.setdefault(number, []).extend(items)
        else:
            fields[number] = value
    # AttributeType: FLOAT 1, INT 2, STRING 3, TENSOR 4, FLOATS 6, INTS 7. Old exporters omit it.
    if not kind:
        kind = next((kind for field, kind in ((2, 1), (3, 2), (4, 3), (5, 4), (7, 6), (8, 7)) if field in fields), 0)
    if kind == 1:
        return name, float(np.frombuffer(fields.get(2, b"\0\0\0\0"), dtype="<f4")[0])
    if kind == 2:
        return name, fields.get(3, 0)
    if kind == 3:
        return name, bytes(fields.get(4, b"")).decode()
    if kind == 4:
        return name, _tensor(fields[5])[1]
    if kind == 6:
        return name, [float(item) for item in fields.get(7, [])]
    if kind == 7:
        return name, list(fields.get(8, []))
    raise ValueError(f"Unsupported ONNX attribute type {kind} for {name!r}")


def _node(data):
    inputs, outputs, op_type, attributes = [], [], "", {}
    for number, _, value in _fields(data):
        if number == 1:
            inputs.append(bytes(value).decode())
        elif number == 2:
            outputs.append(bytes(value).decode())
        elif number == 4:
            op_type = bytes(value).decode()
        elif number == 5:
            key, item = _attribute(value)
            attributes[key] = item
    return op_type, inputs, outputs, attributes


def read_onnx(path):
    """(nodes, initializers, graph input names, graph output names) of an ONNX model file."""
    with open(path, "rb") as file:
        data = memoryview(file.read())
    graph = None
    for number, _, value in _fields(data):
        if number == 7:
            graph = value
    if graph is None:
        raise ValueError(f"{path} is not an ONNX model")
    nodes, initializers, inputs, outputs = [], {}, [], []
    for number, _, value in _fields(graph):
        if number == 1:
            nodes.append(_node(value))
        elif number == 5:
            key, tensor = _tensor(value)
            initializers[key] = tensor
        elif number in (11, 12):
            name = next(bytes(item).decode() for field, _, item in _fields(value) if field == 1)
            (inputs if number == 11 else outputs).append(name)
    return nodes, initializers, [name for name in inputs if name not in initializers], outputs


class OnnxModule(nn.Module):
    """An ONNX graph executed with torch operators, so gradients reach its input."""

    def __init__(self, path):
        super().__init__()
        nodes, initializers, self.inputs, self.outputs = read_onnx(path)
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise ValueError(f"{path} must have one input and one output")
        # Constants become buffers, so they follow the module to its device.
        for op_type, _, outputs, attributes in nodes:
            if op_type == "Constant":
                initializers[outputs[0]] = attributes["value"]
        self.nodes = [node for node in nodes if node[0] != "Constant"]
        self.names = {name: f"tensor_{index}" for index, name in enumerate(initializers)}
        for name, tensor in initializers.items():
            self.register_buffer(self.names[name], tensor)
        for op_type, _, _, _ in self.nodes:
            if not hasattr(self, "_op_" + op_type):
                raise NotImplementedError(f"ONNX operator {op_type} in {path} is not supported")

    def forward(self, x):
        values = {self.inputs[0]: x}
        for name, buffer in self.names.items():
            values[name] = getattr(self, buffer)
        for op_type, inputs, outputs, attributes in self.nodes:
            result = getattr(self, "_op_" + op_type)([values[name] if name else None for name in inputs], attributes)
            values.update(zip(outputs, result if isinstance(result, tuple) else (result,), strict=False))
        return values[self.outputs[0]]

    @staticmethod
    def _op_Conv(inputs, attributes):
        x, weight, bias = (inputs + [None])[:3]
        pads = attributes.get("pads", [0] * (2 * (weight.dim() - 2)))
        half = len(pads) // 2
        if pads[:half] != pads[half:]:
            # ONNX orders pads as all begins, then all ends; F.pad wants pairs from the last dim.
            x = F.pad(x, [value for axis in reversed(range(half)) for value in (pads[axis], pads[axis + half])])
            pads = [0] * len(pads)
        convolution = (F.conv1d, F.conv2d, F.conv3d)[weight.dim() - 3]
        return convolution(
            x, weight.to(x.dtype), None if bias is None else bias.to(x.dtype),
            stride=attributes.get("strides", 1), padding=pads[:half],
            dilation=attributes.get("dilations", 1), groups=attributes.get("group", 1),
        )

    @staticmethod
    def _op_BatchNormalization(inputs, attributes):
        x, scale, bias, mean, variance = inputs[:5]
        return F.batch_norm(x, mean, variance, scale, bias, training=False, eps=attributes.get("epsilon", 1e-5))

    @staticmethod
    def _op_PRelu(inputs, attributes):
        x, slope = inputs
        return torch.where(x >= 0, x, x * slope.to(x.dtype))

    @staticmethod
    def _op_Relu(inputs, attributes):
        return F.relu(inputs[0])

    @staticmethod
    def _op_Add(inputs, attributes):
        return inputs[0] + inputs[1]

    @staticmethod
    def _op_Sub(inputs, attributes):
        return inputs[0] - inputs[1]

    @staticmethod
    def _op_Mul(inputs, attributes):
        return inputs[0] * inputs[1]

    @staticmethod
    def _op_Div(inputs, attributes):
        return inputs[0] / inputs[1]

    @staticmethod
    def _op_Flatten(inputs, attributes):
        x = inputs[0]
        axis = attributes.get("axis", 1) % max(x.dim(), 1)
        return x.reshape(math.prod(x.shape[:axis]), -1)

    @staticmethod
    def _op_Reshape(inputs, attributes):
        x, shape = inputs[:2]
        shape = [x.shape[index] if size == 0 else size for index, size in enumerate(shape.tolist())]
        return x.reshape(shape)

    @staticmethod
    def _op_Gemm(inputs, attributes):
        a, b, c = (inputs + [None])[:3]
        a = a.T if attributes.get("transA", 0) else a
        b = b.T if attributes.get("transB", 0) else b
        result = attributes.get("alpha", 1.0) * (a @ b.to(a.dtype))
        return result if c is None else result + attributes.get("beta", 1.0) * c.to(a.dtype)

    @staticmethod
    def _op_MatMul(inputs, attributes):
        return inputs[0] @ inputs[1].to(inputs[0].dtype)

    @staticmethod
    def _op_Transpose(inputs, attributes):
        return inputs[0].permute(attributes["perm"]) if "perm" in attributes else inputs[0].permute(*reversed(range(inputs[0].dim())))

    @staticmethod
    def _op_Identity(inputs, attributes):
        return inputs[0]

    @staticmethod
    def _op_Dropout(inputs, attributes):
        return inputs[0]
