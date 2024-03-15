from setuptools import setup, find_packages
import os
import sys

if os.path.exists("README.md"):
    README = open("README.md").read()
else:
    README = "No description"

# Machine learning dependencies
if sys.platform == "linux":
    ML_DEPENDENCIES = [
        "fairseq @ git+https://github.com/IAHispano/fairseq",
        "numba",
    ]
else:
    ML_DEPENDENCIES = [
        "fairseq==0.12.2",
        "numba==0.56.4",
    ]

# Other dependencies
OTHER_DEPENDENCIES = [
    "onnxruntime",
    "onnxruntime_gpu==1.15.1",
    "torch==2.1.1",
    "torchcrepe==0.0.21",
    "torchgen>=0.0.1",
    "torch_directml",
    "torchvision==0.16.1",
    "einops",
    "local-attention",
    "matplotlib==3.7.2",
    "ffmpy==0.3.1",
    "edge-tts==6.1.9",
    "pydantic",
]

setup(
    name="rvc_cli",
    version="1.1.1",
    author="Blaise",
    author_email="iahispano0@gmail.com",
    description="RVC CLI enables seamless interaction with Retrieval-based Voice Conversion through commands or HTTP requests.",
    license="Attribution-NonCommercial 4.0 International",
    packages=find_packages(),
    package_data={"rvc_cli": ["*", "logs"]},
    long_description=(README),
    long_description_content_type="text/markdown",
    # requirements
    python_requires=">=3.7, <3.11",
    install_requires=ML_DEPENDENCIES + OTHER_DEPENDENCIES,
    classifiers=[
        "License :: OSI Approved :: Attribution-NonCommercial 4.0 International",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Intended Audience :: Developers",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows :: Windows 10-11",
        "Operating System :: POSIX :: Linux",
    ],
)
