"""Capture and modify layer outputs during target computation."""

from contextlib import AbstractContextManager


class Trace(AbstractContextManager):
    """Keep a layer's output attached to the computation graph."""

    def __init__(self, module, layer, edit_output=None):
        def hook(module, inputs, output):
            if edit_output is not None:
                output = edit_output(output, layer)
            self.output = output
            return output

        self.handle = get_module(module, layer).register_forward_hook(hook)

    def close(self):
        self.handle.remove()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


class TraceDict(dict, AbstractContextManager):
    """Capture named layer outputs and remove all hooks on exit."""

    def __init__(self, module, layers, edit_output=None):
        super().__init__(
            (layer, Trace(module, layer, edit_output))
            for layer in dict.fromkeys(layers)
        )

    def __exit__(self, exc_type, exc_value, traceback):
        for trace in reversed(list(self.values())):
            trace.close()


def set_requires_grad(requires_grad, model):
    """Set gradient tracking for the model parameters."""
    for parameter in model.parameters():
        parameter.requires_grad_(requires_grad)


def get_module(model, name):
    """Find a module by its dotted name."""
    for module_name, module in model.named_modules():
        if module_name == name:
            return module
    raise LookupError(name)


def get_parameter(model, name):
    """Find a parameter by its dotted name."""
    for parameter_name, parameter in model.named_parameters():
        if parameter_name == name:
            return parameter
    raise LookupError(name)
