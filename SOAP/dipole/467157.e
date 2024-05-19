Traceback (most recent call last):
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 79, in __getattr__
    return self[key]
           ~~~~^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 104, in __getitem__
    return self._mapping[key]
           ~~~~~~~~~~~~~^^^^^
KeyError: 'posc'

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/public/home/liuzhipan/yzx/CPUmodel/SOAP-large/dipole/main.py", line 72, in <module>
    out = model(data,mean_proper[property],std_proper[property])
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/yzx/CPUmodel/SOAP-large/dipole/model.py", line 31, in forward
    x = out* data.posc
             ^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch_geometric/data/data.py", line 441, in __getattr__
    return getattr(self._store, key)
           ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/public/home/liuzhipan/wangzx/miniconda3/envs/venv/lib/python3.11/site-packages/torch_geometric/data/storage.py", line 81, in __getattr__
    raise AttributeError(
AttributeError: 'GlobalStorage' object has no attribute 'posc'
