from setuptools import setup, find_packages

setup(
    name='experimental_depth_prediction',
    version='1.0.0',
    author='Francis Ogoke, Peter Pak',
    author_email='oogoke@andrew.cmu.edu',
    description='Code to predict the melt pool depth from thermal images',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        # List your dependencies here
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
