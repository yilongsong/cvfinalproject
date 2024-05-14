from setuptools import setup, find_packages

# read the contents of your README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    lines = f.readlines()

# remove images from README
lines = [x for x in lines if (('.png' not in x) and ('.gif' not in x))]
long_description = ''.join(lines)

setup(
    name="mimicgen_envs",
    packages=[
        package for package in find_packages() if package.startswith("mimicgen_envs")
    ],
    install_requires=[
        "numpy>=1.13.3",
        "h5py",
        "tqdm",
        "imageio",
        "imageio-ffmpeg",
        "gdown",
        "chardet",
        "mujoco==2.3.2",
    ],
    eager_resources=['*'],
    include_package_data=True,
    python_requires='>=3',
    description="MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations",
    author="Ajay Mandlekar",
    url="https://gitlab-master.nvidia.com/srl/mimicgen_environments",
    author_email="amandlek@cs.stanford.edu",
    version="0.1.0",
    long_description=long_description,
    long_description_content_type='text/markdown'
)
