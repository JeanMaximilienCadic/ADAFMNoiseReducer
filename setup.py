from setuptools import setup

setup(
    name='ADAFMNoiseReducer',
    version="1.0.2",
    long_description="Simple noise reducer from ADAFM architecture",
    package_data={
        'ADAFMNoiseReducer': [
            "ADAFMNoiseReducerModel.pth"
        ]
    },
    packages=
    [
        "ADAFMNoiseReducer",
        "ADAFMNoiseReducer/models",
        "ADAFMNoiseReducer/models/modules",
    ],
    include_package_data=True,
    install_requires=[
        "gnutools-python",
        "opencv-python==4.4.0.44",
        "torch==1.6.0",
        "torchvision==0.7.0",
        "numpy==1.19.1",
    ],
    url='https://github.com/JeanMaximilienCadic',
    license='MIT',
    author='Jean Maximilien Cadic',
    python_requires='>=3.6',
    author_email='j.cadic@pm.me',
    description='Simple noise reducer from ADAFM architecture',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)

