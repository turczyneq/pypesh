from setuptools import setup


def parse_requirements(filename):
    with open(filename) as f:
        return f.read().splitlines()


with open("README.md") as f:
    long_description = f.read()

setup(
    name="pypesh",
    version="0.1.1",
    description="Sherwood number in Stokes flow",
    url="https://github.com/turczyneq/pypesh",
    author="Jan Turczynowicz and Radost Waszkiewicz",
    author_email="turczynowicz.jan@wp.pl",
    long_description=long_description,
    long_description_content_type="text/markdown",  # This is important!
    project_urls={
        "Documentation": "https://pypesh.readthedocs.io",
        "Source": "https://github.com/turczyneq/pypesh/",
    },
    license="GNU GPLv3",
    packages=["pypesh"],
    install_requires=parse_requirements("requirements.txt"),
    zip_safe=False,
)
