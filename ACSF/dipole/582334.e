Traceback (most recent call last):
  File "/public/home/liuzhipan/yzx/CPUmodel/ACSF/dipole/main.py", line 72, in <module>
    out = model(data,mean_proper[property],std_proper[property])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/yzx/CPUmodel/ACSF/dipole/model.py", line 26, in forward
    x = x * elez
        ~~^~~~~~
RuntimeError: The size of tensor a (3255) must match the size of tensor b (3275) at non-singleton dimension 1
