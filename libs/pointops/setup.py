import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

# 设置允许不支持的编译器
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
os.environ["NVCC_APPEND_FLAGS"] = "--allow-unsupported-compiler"

(opt,) = get_config_vars("OPT")
if opt is None:
    opt = ""
os.environ["OPT"] = " ".join(
    flag for flag in opt.split() if flag != "-Wstrict-prototypes"
)

src = "src"
sources = [
    os.path.join(root, file)
    for root, dirs, files in os.walk(src)
    for file in files
    if file.endswith(".cpp") or file.endswith(".cu")
]

setup(
    name="pointops",
    version="1.0",
    install_requires=["torch", "numpy"],
    packages=["pointops"],
    package_dir={"pointops": "functions"},
    ext_modules=[
        CUDAExtension(
            name="pointops._C",
            sources=sources,
            extra_compile_args={
                "cxx": ["-g"],
                "nvcc": ["-O2", "--allow-unsupported-compiler"]
            },
            include_dirs=[],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
