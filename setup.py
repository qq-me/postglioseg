from setuptools import setup

setup(
   name='postglioseg',
   version='1.0',
   description='Сегментация постоперационных изображений глиобластомы.',
   author='Никишев Иван Олегович',
   author_email='nkshv2@gmail.com',
   packages=['postglioseg'],  #same as name
   install_requires=['torch', 'numpy', "SimpleITK-SimpleElastix", "hd_glio", "dcm2niix", "gradio", "monai", "pydicom", "highdicom"], #external packages as dependencies
)
