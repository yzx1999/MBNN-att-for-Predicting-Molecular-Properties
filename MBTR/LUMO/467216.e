Traceback (most recent call last):
  File "/public/home/liuzhipan/yzx/CPUmodel/MBTR-large/polar/main.py", line 42, in <module>
    model = OneeleNet(input_channel = input_channel,hidden_channel=hidden_channel).to(device)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1145, in to
    return self._apply(convert)
           ^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 797, in _apply
    module._apply(fn)
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 820, in _apply
    param_applied = fn(param)
                    ^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1143, in convert
    return t.to(device, dtype if t.is_floating_point() or t.is_complex() else None, non_blocking)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: CUDA error: invalid device ordinal
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1.
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.

