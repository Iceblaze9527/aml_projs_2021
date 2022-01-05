import setuptools
from Cython.Build import cythonize
import numpy

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = 'ReverseRemapping',
  ext_modules=[
    Extension('ReverseRemapping',
              sources=['ReverseRemapping.pyx'],
              extra_compile_args=['-O3'])
    ],
  cmdclass = {'build_ext': build_ext}
)

#setuptools.setup(
#    ext_modules=cythonize(["*.pyx"], language_level=3, extra_compile_args=["-O3"]),
#    include_dirs=[numpy.get_include()]
#)
