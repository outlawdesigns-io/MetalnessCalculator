import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="MetalnessCalculator",
    version="1.0.0",
    description="https://medium.com/@luca.ballore/when-heavy-metal-meets-data-science-3fc32e9096fa",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/outlawdesigns-io/MetalnessCalculator",
    author="outlawdesigns.io",
    author_email="j.watson@outlawdesigns.io",
    license="MIT",
    packages=["MetalnessCalculator"],
    include_package_data=True,
    install_requires=["matplotlib","nltk","pandas","scipy"],
    entry_points={
        "console_scripts": [
            "MetalnessCalculator=MetalnessCalculator.__main__:main",
        ]
    },
)
