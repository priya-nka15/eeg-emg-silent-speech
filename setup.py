from setuptools import setup, find_packages

setup(
    name="urf-a-ssr",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26,<2",
        "scipy>=1.13,<2",
        "scikit-learn>=1.5,<2",
        "PyYAML>=6.0,<7",
    ],
    extras_require={
        "dl": ["torch>=2.4,<3"],
        "test": ["pytest>=8.3", "pytest-cov>=5"],
    },
)
