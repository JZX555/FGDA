import torch
from torch.optim.optimizer import Optimizer
from math import sqrt


class Adafactor(Optimizer):
    def __init__(self,
                 params, lr=0.01,
                 betas=(0.0, 0.999),
                 eps=(1e-30, 1e-3),
                 memory_exponent=0.8,
                 grad_clip=1.0,
                 multiply_by_params_scale=True,
                 decay_type='pow'):
        """
        initiate parameters for adafactor optimizer
        :param params: (iterable): iterable of parameters to optimize or dicts defining
                parameter groups
        :param lr: learning rate, optional float scalar
        :param betas: (Tuple[float, float], optional): coefficients used for computing
                running averages of gradient and its square (default: (0.9, 0.999))
                beta1 enables the momentum.
        :param eps: a float, optional term added to the denominator to improve
                numerical stability for [squared_grad, param_scale] (default: 1e-30, 1e-3)
        :param memory_exponent: used for beta2 decay
        :param grad_clip: should be >=1.0 or None for no update clipping
        :param multiply_by_params_scale: If True, then compute absolute_update_scale
                as described above.  If False, let absolute_update_scale be the externally
                supplied learning_rate.
        :param decay_type: 'adam' or 'pow'
        """
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        # more validate initialization
        if not 0.0 <= eps[0]:
            raise ValueError("Invalid epsilon value: {}".format(eps[0]))
        if not 0.0 <= eps[1]:
            raise ValueError("Invalid epsilon value: {}".format(eps[1]))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta1 parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta2 parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= memory_exponent < 1.0:
            raise ValueError("Invalid memory exponent for beta2 decay: {}".format(memory_exponent))
        if not 1.0 <= grad_clip:
            raise ValueError("Invalid clipping threshold: {}".format(grad_clip))

        # dynamic decay rate
        if decay_type not in ["pow", "adam", "constant"]:
            raise ValueError("Invalid decay type for beta2:{},must be 'pow', 'adam', 'constant'.".format(decay_type))

        defaults = dict(lr=lr,
                        betas=betas,
                        eps=eps,
                        memory_exponent=memory_exponent,
                        clipping_threshold=grad_clip,
                        multiply_by_params_scale=multiply_by_params_scale,
                        decay_type=decay_type)
        super(Adafactor, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adafactor, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("multiply_by_params_scale", True)
            group.setdefault("memory_exponent", 0.8)
            group.setdefault("decay_type", "pow")

    def step(self, closure=None):
        """ performs a single step of optimization

        Adafactor updates parameter according to shape(factored or not)
        We initialize
        ```````````````````````````````````
        if var is 2-dimensional: v is always 1-dimensional
          v_r <- zeros([num_rows])
          v_c <- zeros([num_cols])
        if var is 0-dimensional or 1-dimensional:
          v <- zeros(shape(var))
        ```````````````````````````````````
        The update rule is as follows:
        ```````````````````````````
        beta2 = 1 - (step_num + 1) ^ -memory_exponent
        grad_squared = pow(grad,2) + epsilon1
        if var is 2 or more dimensions:
          v_r <- beta2 * v_r + (1 - beta2) * reduce_mean(grad_squared, axis=-1)
          v_c <- beta2 * v_c + (1 - beta2) * reduce_mean(grad_squared, axis=-2)
          v = outer_prod(v_r, v_c) / reduce_mean(v_r)
        if var is 0-dimensional or 1-dimensional:
          v <- beta2 * v + (1 - beta2) * grad_squared
        ```````````````````````````
        See the code for details.

        :param closure: (callable, optional) A closure that reevaluates the model
        :return loss: loss of the optimizer
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adafactor does not support sparse gradient")
                multiply_by_params_scale = group['multiply_by_params_scale']

                state = self.state[p]

                if len(state) == 0:  # initialization:
                    state["step"] = 0
                    if len(grad.shape) >= 2:
                        # factored when grad.shape>=2
                        state['v_r'] = torch.zeros_like(p.data.sum(dim=-1))
                        state['v_c'] = torch.zeros_like(p.data.sum(dim=-2))
                    else:
                        state['v'] = torch.zeros_like(p.data)
                    if group['betas'][0] > 0:
                        # enable momentum
                        state['m'] = torch.zeros_like(p.data)

                state["step"] += 1
                beta1, beta2 = group['betas']
                eps1, eps2 = group['eps']

                grad_spuared = grad.pow(2.0).add_(eps1)
                grad_squared_mean = torch.mean(grad_spuared)

                update_scale = group['lr']
                if multiply_by_params_scale:
                    update_scale *= max(reduce_rms(p.data), eps2)
                else:
                    update_scale *= 0.05

                # beta2 decay if there's any
                if group['decay_type'] == "pow":
                    beta2 = 1.0 - state["step"] ** (- group["memory_exponent"])
                elif group['decay_type'] == "pow":
                    beta2 = beta2 * (1.0 - beta2 ** (state['step'] + 1)) / (1.0 - beta2 ** state['step'])

                scale_correction = grad_squared_mean * 1e-30  # hack from tf.adafactor
                update_scale += scale_correction
                beta2 += scale_correction

                if len(grad.shape) >= 2:
                    # factored along the last 2 dimensions
                    if grad_squared_mean.is_cuda:
                        v_r, v_c = state["v_r"].cuda(), state["v_c"].cuda()
                    else:
                        v_r, v_c = state["v_r"], state["v_c"]
                    v_r.mul_(beta2).add_(1.0 - beta2, torch.mean(grad_spuared, dim=-1))
                    v_c.mul_(beta2).add_(1.0 - beta2, torch.mean(grad_spuared, dim=-2))
                    r_factor = torch.rsqrt(v_r.div(torch.mean(v_r, dim=-1)))
                    c_factor = torch.rsqrt(v_c)
                    U = grad.mul(r_factor.unsqueeze(-1)).mul(c_factor.unsqueeze(-2))
                else:
                    if grad_squared_mean.is_cuda:
                        v = state['v'].cuda()
                    else:
                        v = state['v']
                    v.mul_(beta2).add_(1.0 - beta2, grad_spuared)
                    U = grad.mul(torch.rsqrt(v))

                if group['clipping_threshold'] is not None:
                    U.div_(max(1.0, reduce_rms(U) / (group['clipping_threshold'])))

                subtrahend = U.mul(update_scale)

                if beta1 > 0:
                    # we have momentum
                    if grad_squared_mean.is_cuda:
                        m = state['m'].cuda()
                    else:
                        m = state['m']
                    subtrahend.mul_(1.0 - beta1).add_(beta1, m)

                p.data.add_(-subtrahend)

        return loss


def reduce_rms(x):
    # return torch variable root-mean-square of tensor x
    return sqrt(torch.mean(x.pow(2)))
