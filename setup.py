from os import path

from setuptools import find_packages, setup

dirname = path.abspath(path.dirname(__file__))
with open(path.join(dirname, 'README.md')) as f:
    long_description = f.read()

setup(
    name='featuretoolsOnSpark',
    version='0.1.4',
    packages=find_packages(),
    description='A simplified version of featuretools for Spark',
    license='MIT',
    url='https://github.com/giantcroc/featuretoolsOnSpark',
    author='giantcroc',
    author_email='1204449533@qq.com',
    classifiers=[
         'Development Status :: 2 - Pre-Alpha',
         'Intended Audience :: Developers',
         'Programming Language :: Python :: 2.7',
         'Programming Language :: Python :: 3',
         'Programming Language :: Python :: 3.5',
         'Programming Language :: Python :: 3.6',
         'Programming Language :: Python :: 3.7'
    ],
    install_requires=open('requirements.txt').readlines(),
    python_requires='>=2.7, <4',
    keywords='feature engineering data science machine learning',
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown'
)
