import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors


def get_dtype(dtype_str):
    if isinstance(dtype_str, str):
        return {
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
            'float32': torch.float32,
            'fp16': torch.float16,
            'fp32': torch.float32,
            'bf16': torch.bfloat16,
        }[dtype_str]
    elif isinstance(dtype_str, torch.dtype):
        return dtype_str
    else:
        raise ValueError(f"Invalid dtype: {dtype_str}")

def make_master_params(model_params):
    master_params = _flatten_dense_tensors([param.detach().float() for param in model_params])
    master_params = nn.Parameter(master_params)
    master_params.requires_grad = True
    return [master_params]

def model_params_to_master_params(model_params, master_params):
    master_params[0].detach().copy_(
        _flatten_dense_tensors([param.detach().float() for param in model_params])
    )

def master_params_to_model_params(model_params, master_params):
    for param, master_param in zip(
        model_params, _unflatten_dense_tensors(master_params[0].detach(), model_params)
    ):
        param.detach().copy_(master_param)

def model_grads_to_master_grads(model_params, master_params):
    grads = [p.grad.data.detach().float() if p.grad is not None else torch.zeros_like(p.data).float() 
             for p in model_params]
    master_params[0].grad = _flatten_dense_tensors(grads)

class InflatAllOptimizerWrapper:
    """
    包装器：实现 'inflat_all' 混合精度策略。
    它维护一个扁平化的 FP32 Master Parameter，并在 step 前后自动处理
    Model Params (FP16/BF16) 与 Master Params (FP32) 之间的同步与梯度缩放。
    """
    def __init__(
        self, 
        optimizer_cls, 
        optimizer_kwargs, 
        model_params, 
        dtype=torch.float16, 
        init_scale_log=20.0,
        scale_growth=1e-3,
        min_scale_log=0.0
    ):
        self.dtype = get_dtype(dtype)
        self.model_params = list(model_params) # 保持引用
        self.master_params = make_master_params(self.model_params)
        self.optimizer = optimizer_cls(self.master_params, **optimizer_kwargs)
        
        # 3. 混合精度相关状态
        self.log_scale = init_scale_log
        self.scale_growth = scale_growth
        self.min_scale_log = min_scale_log
        self._is_finite = True

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def zero_grad(self):
        self.optimizer.zero_grad()
        for param in self.model_params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def backward(self, loss):
        """
        处理 Loss Scaling 并反向传播
        """
        if self.dtype == torch.float16:
            scaled_loss = loss * (2 ** self.log_scale)
            scaled_loss.backward()
        else:
            loss.backward()

    def _sync_grads(self):
        """内部方法：将模型梯度同步到 Master Params 并进行 Unscale"""
        model_grads_to_master_grads(self.model_params, self.master_params)
        
        if self.dtype == torch.float16:
            # Unscale gradient
            scale = 2 ** self.log_scale
            self.master_params[0].grad.mul_(1.0 / scale)

    def clip_grad_norm(self, max_norm, norm_type=2):
        """
        并在 Master Params 上执行梯度裁剪。
        必须在 step() 之前手动调用 (如果需要裁剪)。
        注意：调用此函数会触发梯度同步。
        """
        # 确保梯度已同步且 unscaled
        self._sync_grads()
        
        # 检查 NaN/Inf (在 unscale 之后检查)
        self._check_overflow()
        
        if self._is_finite:
            return torch.nn.utils.clip_grad_norm_(self.master_params, max_norm, norm_type=norm_type)
        else:
            return torch.tensor(float('inf'))

    def _check_overflow(self):
        """检查 Master Grads 是否有 Inf/NaN"""
        if not torch.isfinite(self.master_params[0].grad).all():
            self._is_finite = False
        else:
            self._is_finite = True

    def step(self):
        """
        执行优化步。
        包含：同步梯度(若未裁剪)、检查溢出、更新参数、同步回模型、更新 Scale。
        """
        # 如果之前没有调用 clip_grad_norm，梯度可能还没同步过来
        if self.master_params[0].grad is None: 
            self._sync_grads()
            self._check_overflow()

        # 根据是否溢出决定是否更新
        if self.dtype == torch.float16:
            if self._is_finite:
                self.optimizer.step()
                # 将更新后的 Master Params 同步回 Model Params
                master_params_to_model_params(self.model_params, self.master_params)
                # 增加 Scale
                self.log_scale += self.scale_growth
            else:
                print(f'\n[InflatAll] Warning: NaN/Inf detected. Skipping step. Reducing scale from {self.log_scale:.2f} to {self.log_scale - 1:.2f}')
                self.log_scale -= 1.0
        else:
            # 非 FP16 模式 (如 BF16 或 FP32 使用 inflat 策略)
            if self._is_finite:
                self.optimizer.step()
                master_params_to_model_params(self.model_params, self.master_params)
            else:
                print('\n[InflatAll] Warning: NaN/Inf detected in gradients. Skipping update.')

        return self._is_finite

    def state_dict(self):
        """包含优化器状态和 Scale 状态"""
        state = self.optimizer.state_dict()
        state['custom_log_scale'] = self.log_scale
        return state

    def load_state_dict(self, state_dict):
        self.log_scale = state_dict.pop('custom_log_scale', 20.0)
        self.optimizer.load_state_dict(state_dict)
        master_params_to_model_params(self.model_params, self.master_params)

    def get_scale(self):
        if self.dtype == torch.float16:
            return 2 ** self.log_scale
        return 1.0