from setuptools import setup

setup(name='MILPy',
      version='0.1.0',
      description='MILPy is a collection of miltiple instance learning (MIL) algorithms implemented in the python programming language. Currently only few algorithms are implemented: Bayesian KNN, Citation KNN, SIL, MISVM, miSVM, NSK, STK, MissSVM, MICA, sMIL, stMIL, sbMIL, miGraph.',
      url='https://github.com/manuSrep/MILPy.git',
      author='Manuel Tuschen',
      author_email='Manuel_Tuschen@web.de',
      license='GPL3 License',
      packages=['MILPy'],
      install_requires=["scipy", "numpy","scikit-learn","cvxopt", "progressbar2","miscpy"],
      zip_safe=False,
      )
