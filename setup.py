from setuptools import setup, find_packages

setup(
    name='diffaero',
    version='0.1',
    packages=["diffaero", "diffaero.env", "diffaero.algo", "diffaero.network", "diffaero.utils", "diffaero.script"],
    package_dir={"diffaero": "."},
    install_requires=[
        'torch>=2.0.0',
        'tensordict',
        'taichi',
        'tqdm',
        'hydra-core',
        'hydra-joblib-launcher',
        'hydra_colorlog',
        'hydra-optuna-sweeper',
        'welford_torch',
        'line_profiler',
        'tensorboard',
        'tensorboardX',
        'torch-tb-profiler',
        'wandb',
        'gpustat',
        'opencv-python',
'open3d',
        'numpy',
        'moviepy==1.0.3',
        'imageio',
        'imageio-ffmpeg',
        'matplotlib',
        'onnx',
        'onnxruntime'
    ],
    author='Xinhong Zhang',
    author_email='xhzhang@bit.edu.cn',
    description='',
    url='https://github.com/flyingbitac/diffaero'
)
