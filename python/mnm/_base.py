def set_module(module):
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator
