from setuptools import setup

setup(
    name='hypt',
    version='0.1.0',
    description='Simple hyperparameter tuning in Python',
    author='João Bravo',
    url='https://github.com/ajoo/hypt',
    packages=['hypt'],
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
    ],
)
