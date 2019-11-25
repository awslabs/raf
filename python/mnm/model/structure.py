from mnm._core.model import Model
from mnm._core.script import script_model as script

class Sequential(Model):

    # pylint: disable=attribute-defined-outside-init
    def build(self, *args):
        self.num_layers = len(args)

        for idx, layer in enumerate(args):
            setattr(self, "layer" + str(idx), layer)
    # pylint: enable=attribute-defined-outside-init

    @script
    def forward(self, x):
        for idx in range(self.num_layers):
            layer = getattr(self, "layer" + str(idx))
            x = layer(x)

        return x
