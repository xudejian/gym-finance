from setuptools import setup, find_packages

setup(
    name='gym_finance',
    version='0.0.1',
    packages=find_packages(),

    author='DJ',
    author_email='dxu2050@gmail.com',
    license='MIT',

    install_requires=[
        'gymnasium>=0.29.0',
        'numpy>=1.16.4',
        'pandas>=0.24.2',
    ]
)
