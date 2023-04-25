# -*- coding: utf-8 -*-

"""Setup module"""

import setuptools

setuptools.setup(
    name='topML',
    version='0.1.0',
    author='Seth Moore',
    author_email='mooreseth@uchicago.edu',
    description='Topological Data Analysis Machine Learning Framework - designed for molecular data',

    url='https://github.com/mooresethmoore/topML',
    license='MIT',
    packages=['topML'],
    install_requires=['requests'],
) #    long_description=long_description,
    #long_description_content_type="text/markdown",


### install with
# pip install git+https://github.com/mooresethmoore/topML.git