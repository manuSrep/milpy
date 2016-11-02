from setuptools import setup

setup(name='MILPy',
      version='0.1.0',
      description='MILPy is a collection of miltiple instance learning (MIL) algorithms implemented in the python programming language. \n '
                  'Currently only few algoritms are implemented: \n'
                  '* Bayesian KNN \n'
                  '* Citation KNN \n'
                  '* miGraph \n',
      url='https://github.com/manuSrep/MILPy.git',
      author='Manuel Tuschen',
      author_email='Manuel_Tuschen@web.de',
      license='GPL3 License',
      packages=['MILPy'],
      install_requires=["scipy", "cvxopt", "numpy","progressbar2","miscpy"],
      zip_safe=False,
      )
