import numpy
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

ext_modules = [
    Extension("pltsurf", ["cython_modules/pltsurf.pyx"],
              libraries=["m"],  # Unix-like specific
              include_dirs=[numpy.get_include()]
              ),
    Extension("expsim", ["cython_modules/expsim.pyx"],
              libraries=["m"],  # Unix-like specific
              include_dirs=[numpy.get_include()]
              )
]

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='OptimInventory',
    version='2.0',
    packages=[''],
    ext_modules=cythonize(ext_modules, **ext_options),
    url='',
    license='',
    author='tomislav',
    author_email='tbazina@gmail.com',
    description='Optimizacija upravljanja zalihama dobavljackih lanaca'
)
