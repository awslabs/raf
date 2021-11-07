# pylint: disable=invalid-name, missing-function-docstring, too-many-instance-attributes, too-many-locals, too-many-statements
"""LANS optimizer."""
import numpy as np

from mnm._core.core_utils import get_chained_attr
from mnm._core.ndarray import array, ndarray
from mnm.model import trace, Model, trace_mutate_attr
from mnm.model.trace import _get_func_inputs
from mnm._op import sym as _op
from .optim import with_autodiff
from .utils import has_grad


def with_lans(lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0.01):
    """ Optimizer : LANS
    # References
    - Accelerated Large Batch Optimization of BERT Pretraining in 54 minutes.
      http://arxiv.org/abs/2006.13484

    Parameters:
    -----------
    lr: float (optional)
        Learning rate. Default: 1e-3

    betas: Tuple[float, float] (optional)
        Coefficients used for computing running averages of gradient and its square.
        Default: (0.9, 0.999)

    eps: float (optional)
        Term added to the denominator to improve numerical stability. Default: 1e-6

    weight_decay: float (optional)
        Weight decay (L2 penalty). Default: 0.01

    Returns
    ret : function
        The wrapper which wraps a model with LANS
    """
    def decorator(model):
        class LANSWrapper(Model):
            """LANS wrapper model

            Parameters
            ----------
            model: the forward model
            """
            # pylint: disable=attribute-defined-outside-init, protected-access
            def build(self, model):
                self.model = model
                self.ad_model = with_autodiff(model)
                # constants used by LANS
                self.lr = array(lr, dtype='float32')
                self.eps = array(eps, dtype='float32')
                self.weight_decay = array(weight_decay, dtype='float32')
                self.beta1 = array(betas[0], dtype='float32')
                self.beta2 = array(betas[1], dtype='float32')
                self.beta3 = array(1.0 - betas[0], dtype='float32')
                self.beta4 = array(1.0 - betas[1], dtype='float32')
                # other constants
                self.one = array(1.0, dtype='float32')
                self.zero = array(0.0, dtype='float32')
                # mutable params: global step, and running averages
                device = None
                self.params = {}
                for name, x in self.model.state().items():
                    if x.requires_grad is True:
                        if device is None:
                            device = x.device
                        else:
                            assert device == x.device
                        assert isinstance(x, ndarray), "Only `mnm.ndarray` can be optimized!"
                        npa = np.zeros(x.shape, dtype=x.dtype)
                        m_i = array(npa, device=device, name=f'{name}.m')
                        v_i = array(npa, device=device, name=f'{name}.v')
                        setattr(self, f'{name}.m', m_i)
                        setattr(self, f'{name}.v', v_i)
                        self.params[x._ndarray__handle] = (name, x, m_i, v_i)  # pylint: disable=protected-access
                assert device is not None
                self.step = array(0.0, dtype='float32', device=device, name='step')


            @trace
            def forward(self, dy, *args, **kwargs):
                y, dxs = self.ad_model(dy, *args, **kwargs)
                record = self.ad_model._internal(dy, *args, **kwargs)
                inputs = _get_func_inputs(record, [dy, *args], kwargs)
                inputs = inputs[1:]  # remove dy
                # update step
                next_step = _op.add(self.step, self.one, out=self.step)
                trace_mutate_attr(self, 'step', next_step)
                # apply LANS for each param
                for i, param in enumerate(inputs):
                    dxi = dxs[i] if len(inputs) > 1 else dxs
                    if param in self.params and has_grad(dxi):
                        name, x, m, v = self.params[param]
                        if "float" in x.dtype:
                            next_m, next_v, next_x = self._lans(x, dxi, m, v, next_step)
                            param_model = get_chained_attr(self.model, name.split('.')[:-1])
                            trace_mutate_attr(param_model, name.split('.')[-1], next_x)
                            trace_mutate_attr(self, f'{name}.m', next_m)
                            trace_mutate_attr(self, f'{name}.v', next_v)
                return y

            def _l2_norm(self, x):  # pylint: disable=no-self-use
                y = _op.l2norm(x)
                return y

            def _lans(self, p, g, m, v, step):  # pylint: disable=too-many-arguments
                # normalize grad
                g_norm = self._l2_norm(g)
                scaled_g = _op.where(
                    _op.greater(g_norm, self.zero),
                    _op.divide(g, _op.add(g_norm, self.eps)),
                    g
                )
                # Adam
                next_m = _op.add(
                    _op.multiply(self.beta1, m),
                    _op.multiply(self.beta3, scaled_g),
                    out=m
                )
                squared_g = _op.multiply(scaled_g, scaled_g)
                next_v = _op.add(
                    _op.multiply(self.beta2, v),
                    _op.multiply(self.beta4, squared_g),
                    out=v
                )
                # bias correction
                next_m_unbiased = _op.divide(
                    next_m, _op.subtract(self.one, _op.power(self.beta1, step)))
                next_v_unbiased = _op.divide(
                    next_v, _op.subtract(self.one, _op.power(self.beta2, step)))
                # calculate updates
                denom = _op.add(_op.sqrt(next_v_unbiased), self.eps)
                p_norm = self._l2_norm(p)
                scaled_p = _op.multiply(self.weight_decay, p)

                ratios1 = _op.divide(next_m_unbiased, denom)
                tmp1 = _op.add(ratios1, scaled_p)
                tmp1_norm = self._l2_norm(tmp1)
                ratio1 = _op.where(
                    _op.greater(p_norm, self.zero),
                    _op.where(
                        _op.greater(tmp1_norm, self.zero),
                        _op.divide(p_norm, tmp1_norm),
                        self.one),
                    self.one
                )
                update1 = _op.multiply(
                    _op.multiply(_op.multiply(self.lr, ratio1), self.beta1), tmp1)

                ratios2 = _op.divide(scaled_g, denom)
                tmp2 = _op.add(ratios2, scaled_p)
                tmp2_norm = self._l2_norm(tmp2)
                ratio2 = _op.where(
                    _op.greater(p_norm, self.zero),
                    _op.where(
                        _op.greater(tmp2_norm, self.zero),
                        _op.divide(p_norm, tmp2_norm),
                        self.one),
                    self.one
                )
                update2 = _op.multiply(
                    _op.multiply(_op.multiply(self.lr, ratio2), self.beta3), tmp2)
                # apply update
                next_p = _op.subtract(p, update1)
                next_p = _op.subtract(next_p, update2, out=p)
                return next_m, next_v, next_p

        return LANSWrapper(model)
    return decorator
