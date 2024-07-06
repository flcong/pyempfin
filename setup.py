from setuptools import setup

setup(name='pyempfin',
      version='0.1.3',
      description='Helper functions for empirical finance research',
      author='Francis Cong',
      license='MIT',
      packages=['pyempfin'],
      install_requires=[
          'numpy',
          'pandas',
          'numba',
          'tabulate',
          'joblib',
          'scipy'
      ],
      zip_safe=False)
