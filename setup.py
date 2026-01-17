from setuptools import setup, find_packages 
 
setup( 
    name="wifi-human-detection", 
    version="1.0.0", 
    packages=find_packages(), 
    install_requires=[ 
        "torch^>=1.12.0", 
        "numpy^>=1.21.0", 
    ], 
) 
