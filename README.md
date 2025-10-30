## System Requirements
Tested with a Nvidia A100 80GB GPU in the Nova cluster
    - CUDA 12.8
    - RAM 256GB

If needed using different version of cuda
Go to `pyproject.toml' and replace ??? into [128, 126, 118].

```
[tool.uv.sources]
torch = [
  { index = "pytorch-cu???", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu???", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu???"
url = "https://download.pytorch.org/whl/cu???"
explicit = true
```

Or for a cpu-only environment

```
[tool.uv.sources]
torch = [
  { index = "pytorch-cpu" },
]
torchvision = [
  { index = "pytorch-cpu" },
]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

## Run the program
uv run main.py --test --model_name="bert-base-uncased-82" (Please run in /work/classtmp/htlee/coms5790-project2 on Nova)


