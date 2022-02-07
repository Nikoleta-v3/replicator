from setuptools import find_packages, setup

# Read in the version number
exec(open("src/replicator/version.py", "r").read())

with open("requirements.txt") as f:
    reqs = f.read().strip().split("\n")

setup(
    name="replicator",
    version=__version__,
    install_requires=reqs,
    author="Nikoleta Glynatsi",
    author_email=("glynatsi@evolbio.mpg.de"),
    packages=find_packages("src"),
    package_dir={"": "src"},
    description="A package for replicator dynamics.",
)
