"""Setup script."""

import os
import subprocess
import sys


_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


_SETUP_REQUIREMENTS = ["setuptools", "wheel"]

_WINDOWS_REQUIREMENTS = {
    "torch": "https://download.pytorch.org/whl/torch_stable.html",
}


def _preinstall_requirement(requirement, options=None):
    args = ["pip", "install", requirement, *(options or [])]
    return_code = subprocess.call(args)
    if return_code != 0:
        raise RuntimeError(f"{requirement} installation failed")


with open(os.path.join(_ROOT_DIR, "llama3", "version.py")) as f:
    tmp = {}
    exec(f.read(), tmp)
    _VERSION = tmp["VERSION"]
    del tmp

with open(os.path.join(_ROOT_DIR, "README.md"), encoding="utf-8") as f:
    _README = f.read()

# Manually preinstall setup requirements since build system specification in
# pyproject.toml is not reliable. For example, when NumPy is preinstalled,
# NumPy extensions are compiled with the latest compatible NumPy version
# rather than the one available on the system. If the two NumPy versions
# do not match, a runtime error is raised
for requirement in _SETUP_REQUIREMENTS:
    _preinstall_requirement(requirement)

# Windows-specific requirement
if sys.platform in ["cygwin", "win32", "windows"]:
    for requirement, url in _WINDOWS_REQUIREMENTS.items():
        _preinstall_requirement(requirement, options=["-f", url])


from setuptools import find_packages, setup  # noqa: E402


setup(
    name="llama3",
    version=_VERSION,
    description="A single-file implementation of LLaMA 3, with support for jitting, KV caching and prompting",
    long_description=_README,
    long_description_content_type="text/markdown",
    author="Luca Della Libera",
    author_email="luca.dellalib@gmail.com",
    url="https://github.com/lucadellalib/llama3",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Environment :: Console",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: META LLAMA 3 COMMUNITY LICENSE AGREEMENT",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    license="META LLAMA 3 COMMUNITY LICENSE AGREEMENT",
    keywords=["LLaMA 3", "PyTorch"],
    platforms=["OS Independent"],
    include_package_data=True,
    install_requires=["torch"],
    extras_require={"test": ["tiktoken"], "all": ["tiktoken", "torch"]},
    python_requires=">=3.8",
)
