# Runs the installation. See the following for more detail:
# https://docs.python.org/3/distutils/setupscript.html

from setuptools import find_packages, setup

# Avoids duplication of requirements
with open("requirements.txt") as file:
    requirements = file.read().splitlines()

setup(
    name="synergen",
    author="Felix Pei",
    author_email="felp8484@gmail.com",
    description="simulates neural data from dynamical systems",
    url="https://github.com/felixp8/synergen",
    install_requires=requirements,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    version="0.0.1",
)