import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="humpback",
    version="0.0.1",
    author="Jakub Sieroń",
    author_email="j.sieron@student.uw.edu.pl",
    description="Some tools for exploring and filtering large databases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qwymierne/humpback",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)