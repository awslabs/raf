from collections import OrderedDict

def to_arg_list(args: OrderedDict):
    res = []
    for name, arg in args.items():
        to_append = f'{name}: {arg.type}'
        if arg.default is not None:
            to_append = to_append + '=' + str(arg.default)
        res.append(to_append)
    return res

def to_impl_param_list(args: OrderedDict, attrs: OrderedDict):
    res = [name for name, _ in args.items()] + [f'{name}={name}' for name, _ in attrs.items()]
    return res
