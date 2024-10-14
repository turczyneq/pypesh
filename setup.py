from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(name='pypesh',
      version='0.0.0',
      description='Sherwood number in Stokes flow',
      url='https://github.com/turczyneq/pypesh',
      author='Jan Turczynowicz and Radost Waszkiewicz',
      author_email='turczynowicz.jan@wp.pl',
      long_description=long_description,
      long_description_content_type='text/markdown',  # This is important!
    #   project_urls = {
    #       'Documentation': 'https://pygrpy.readthedocs.io',
    #       'Source': 'https://github.com/RadostW/PyGRPY/'
    #   },
      license='GNU GPLv3',
      packages=['pypesh'],
      zip_safe=False)