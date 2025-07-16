from setuptools import setup, find_packages

setup(
    name='face-emotion-recognition',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'opencv-python',
        'tensorflow',
        'numpy',
        'keras',
        'scikit-learn'
    ],
    entry_points={
        'console_scripts': [
            'realtimedetection=realtimedetection:main',
            'train_model=train_model:main',
        ],
    },
    author='Your Name',
    author_email='your.email@example.com',
    description='A project for real-time face emotion recognition using deep learning.',
    license='MIT',
    keywords='face emotion recognition deep learning',
    url='https://github.com/yourusername/face-emotion-recognition',
)