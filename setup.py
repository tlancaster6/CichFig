from distutils.core import setup

setup(name='CichFig',
      version='1.0.0',
      description='standalone plotting suite for the cichlid bower project',
      author='Tucker Lancaster',
      author_email='tuckerlancaster@gmail.com',
      requires=['seaborn', 'scikit-image', 'scikit-learn', 'matplotlib', 'numpy', 'pandas', 'scipy']
      )
