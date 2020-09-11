from setuptools import setup,find_packages
from os import path

current_directory = path.abspath(path.dirname(__file__))


def get_install_requirements():
    requirements_path = path.join(current_directory, "requirements.txt")
    with open(requirements_path, encoding="utf-8") as fp:
        return fp.read().splitlines()


setup(
    name="BOML",
    version="0.1.a1",
    packages=find_packages(),
    url="https://github.com/dut-media-lab/BOML",
    license="MIT",
    install_requires=get_install_requirements(),
    author="Yaohua Liu, Risheng Liu",
    author_email="liuyaohua@mail.dlut.edu.cn",
    description="A Bilevel Optimizer Library in Python for Meta Learning",
)
