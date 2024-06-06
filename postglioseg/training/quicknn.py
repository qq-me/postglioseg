"""Генераторы блоков"""
from typing import Optional
from collections.abc import Sequence, Iterable, Callable
import torch
def conv(
    in_channels: int,
    out_channels: int,
    kernel_size,
    stride = 1,
    padding = 0,
    dilation = 1,
    groups=1,
    bias:bool = True,
    batch_norm: bool = False,
    dropout: Optional[float] = None,
    act: Optional[torch.nn.Module] = None,
    pool: Optional[torch.nn.Module] = None,
    ndim: int = 2,
    custom_op = None,
):
    """Свёрточный блок с пакетной нормализацией и исключением"""
    if ndim == 1:
        Convnd = torch.nn.Conv1d
        BatchNormnd = torch.nn.BatchNorm1d
        Dropoutnd = torch.nn.Dropout1d
    elif ndim == 2:
        Convnd = torch.nn.Conv2d
        BatchNormnd = torch.nn.BatchNorm2d
        Dropoutnd = torch.nn.Dropout2d
    elif ndim == 3:
        Convnd = torch.nn.Conv3d
        BatchNormnd = torch.nn.BatchNorm3d
        Dropoutnd = torch.nn.Dropout3d
    else: raise NotImplementedError
    if custom_op is not None: Convnd = custom_op

    # Список модулей со 3D свёрточным модулем
    modules: list[torch.nn.Module] = [
        Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    ]

    # Функция пулинга
    if pool is not None: modules.append(pool)

    # Функция активации
    if act is not None: modules.append(act)

    # Пакетная нормализация
    if batch_norm is True: modules.append(BatchNormnd(out_channels))

    # Исключение
    if dropout is not None and dropout != 0: modules.append(Dropoutnd(dropout))

    # Возвращается Sequential c распакованным списком модулей
    return torch.nn.Sequential(*modules)

def convt(
    in_channels: int,
    out_channels: int,
    kernel_size,
    stride = 1,
    padding = 0,
    output_padding = 0,
    groups=1,
    bias:bool = True,
    dilation = 1,
    batch_norm: bool = False,
    dropout: Optional[float] = None,
    act: Optional[torch.nn.Module] = None,
    pool: Optional[torch.nn.Module] = None,
    ndim: int = 2,
    custom_op = None,
):
    """Свёрточный блок с пакетной нормализацией и исключением"""
    if ndim == 1:
        Convnd = torch.nn.ConvTranspose1d
        BatchNormnd = torch.nn.BatchNorm1d
        Dropoutnd = torch.nn.Dropout1d
    elif ndim == 2:
        Convnd = torch.nn.ConvTranspose2d
        BatchNormnd = torch.nn.BatchNorm2d
        Dropoutnd = torch.nn.Dropout2d
    elif ndim == 3:
        Convnd = torch.nn.ConvTranspose3d
        BatchNormnd = torch.nn.BatchNorm3d
        Dropoutnd = torch.nn.Dropout3d
    else: raise NotImplementedError
    if custom_op is not None: Convnd = custom_op

    # Список модулей со 3D свёрточным модулем
    modules: list[torch.nn.Module] = [
        Convnd(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            bias=bias,
            dilation=dilation,
        )
    ]

    # Функция активации
    if act is not None: modules.append(act)

    # Функция пулинга
    if pool is not None: modules.append(pool)

    # Пакетная нормализация
    if batch_norm is True: modules.append(BatchNormnd(out_channels))

    # Исключение
    if dropout is not None and dropout != 0: modules.append(Dropoutnd(dropout))

    # Возвращается Sequential c распакованным списком модулей
    return torch.nn.Sequential(*modules)


def linear(
    in_features: Optional[int],
    out_features: int,
    bias: bool = True,
    batch_norm: bool = False,
    dropout: Optional[float] = None,
    act: Optional[torch.nn.Module] = None,
    flatten: bool = False,
    lazy=False,
):
    """Линейный блок с пакетной нормализацией и исключением"""
    # Список модулей со 3D свёрточным модулем
    if lazy: modules:list[torch.nn.Module]  = [torch.nn.LazyLinear(out_features, bias)]
    else: 
        if in_features is None: raise ValueError("in_features must be set")
        modules:list[torch.nn.Module]  = [torch.nn.Linear(in_features, out_features, bias)]

    if flatten: modules.insert(0, torch.nn.Flatten())

    # Функция активации
    if act is not None: modules.append(act)

    # Пакетная нормализация
    if batch_norm is True: modules.append(torch.nn.BatchNorm1d(out_features))

    # Исключение
    if dropout is not None and dropout != 0: modules.append(torch.nn.Dropout(dropout))

    # Возвращается Sequential c распакованным списком модулей
    return torch.nn.Sequential(*modules)


def seq(layers, *args):
    """Создаёт слой со слоями из списка"""
    modules = []
    if isinstance(layers, Sequence): modules.extend(layers)
    else: modules.append(layers)
    modules.extend(args)
    return torch.nn.Sequential(*modules)


def block(
    modules: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module],
    out_channels: Optional[int] = None,
    batch_norm: bool = False,
    dropout: Optional[float] = None,
    act: Optional[torch.nn.Module] = None,
    pool: Optional[torch.nn.Module] = None,
    ndim = 2,
):
    if isinstance(modules, torch.nn.Module): modules = [modules]
    if isinstance(modules, tuple): modules = list(modules)

    if ndim == 1:
        BatchNormnd = torch.nn.BatchNorm1d
        Dropoutnd = torch.nn.Dropout1d
    elif ndim == 2:
        BatchNormnd = torch.nn.BatchNorm2d
        Dropoutnd = torch.nn.Dropout2d
    elif ndim == 3:
        BatchNormnd = torch.nn.BatchNorm3d
        Dropoutnd = torch.nn.Dropout3d
    else: raise NotImplementedError


    # Функция активации
    if act is not None: modules.append(act)

    # Функция пулинга
    if pool is not None: modules.append(pool)

    # Пакетная нормализация
    if batch_norm is True:
        if out_channels is None: raise ValueError("out_channels must be specified for batch_norm")
        modules.append(BatchNormnd(out_channels))

    # Исключение
    if dropout is not None and dropout != 0: modules.append(Dropoutnd(dropout))

    # Возвращается Sequential c распакованным списком модулей
    return torch.nn.Sequential(*modules)
