from os import path

from setuptools import find_packages, setup

dirname = path.abspath(path.dirname(__file__))
with open(path.join(dirname, 'README.md')) as f:
    long_description = f.read()

setup(
    name='featuretoolsOnSpark',
    version='0.1.0',
    packages=find_packages(),
    description='A simplified version of featuretools for Spark',
    license='MIT',
    author='giantcroc',
    classifiers=[
         'Development Status :: 1 - Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.5',
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.7'
    ],
    python_requires='>=2.7, <4',
    keywords='feature engineering data science machine learning',
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
