from setuptools import setup, find_packages

setup(
    name='sedo',
    version='0.1.0',
    description='Social Entropy Diffusion Optimization Algorithm',
    author='safe049',
    author_email='safe049@163.com',
    url='https://github.com/safe049/sedo',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ],
    python_requires='>=3.8',
)