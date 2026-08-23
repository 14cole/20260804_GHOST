#!/usr/bin/env python3

from setuptools import find_packages, setup


setup(
    name="cem-tools",
    version="0.2.0",
    description="Headless and Qt tools for solver-compatible GRIM datasets",
    python_requires=">=3.6",
    packages=find_packages(include=("cem_tools", "cem_tools.*")),
    install_requires=[
        'dataclasses==0.8; python_version == "3.6"',
        'numpy==1.19.5; python_version == "3.6"',
        'numpy==1.21.6; python_version == "3.7"',
        'numpy>=1.24; python_version >= "3.8"',
    ],
    extras_require={
        "gui": [
            'PySide2==5.15.2; python_version < "3.8"',
            'PySide6>=6.5; python_version >= "3.8"',
        ],
    },
    entry_points={
        "console_scripts": [
            "cem-tools=cem_tools.cli:main",
            "cem-tools-gui=cem_tools.gui:run_gui",
        ]
    },
)
