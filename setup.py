from setuptools import setup, find_packages

setup(
    name="llm-bench",
    version="0.1.0",
    description="Compare 20+ local LLMs on your hardware — speed, quality, and memory before downloading",
    author="llm-bench contributors",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "streamlit>=1.35.0",
        "plotly>=5.20.0",
        "pandas>=2.1.0",
        "numpy>=1.26.0",
        "psutil>=5.9.0",
        "pyyaml>=6.0",
        "tqdm>=4.66.0",
        "rich>=13.7.0",
        "click>=8.1.7",
    ],
    extras_require={
        "gpu": [
            "torch>=2.3.0",
            "transformers>=4.42.0",
            "accelerate>=0.30.0",
            "bitsandbytes>=0.43.0",
            "datasets>=2.20.0",
        ],
        "gguf": ["llama-cpp-python>=0.2.82"],
        "dev": ["pytest>=8.0.0", "pytest-cov>=5.0.0"],
    },
    entry_points={
        "console_scripts": [
            "llm-bench=llm_bench.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
