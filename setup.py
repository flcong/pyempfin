from setuptools import setup

setup(name='pyempfin',
      version='0.1',
      description='Helper functions for empirical finance research',
      # url='http://github.com//funniest',
      author='Francis Cong',
      # author_email='flyingcircus@example.com',
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
