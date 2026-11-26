from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="spsa-paper-implementation",
    version="1.0.0",
    author="Ananth Shyam, Rama Narasimha, Mahan Bhimireddy",
    description="SPSA for layer-wise training of deep networks (Wulff et al., ICANN 2018) - PyTorch reproduction",
    long_description=Path(__file__).with_name("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["spsa_nn", "spsa_nn.*"]),
    python_requires=">=3.9",
    install_requires=["numpy>=1.22", "torch>=2.0", "matplotlib>=3.5"],
    extras_require={"test": ["pytest>=7.0"]},
    license="MIT",
)
