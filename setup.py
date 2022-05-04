from setuptools import setup, find_packages

setup(
    name="clip_benchmark-cat-state",
    version="0.0.1",
    author="cat-state",
    author_email="cat-state@tutanota.com",
    description="Compute retrieval benchmark for multimodal CLIP-like embeddings",
    url="https://github.com/cat-state/clip_benchmark",
    packages=find_packages(),
    python_requires=">=3.7",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
