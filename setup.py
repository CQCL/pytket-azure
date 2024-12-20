# Copyright 2020-2024 Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil

from setuptools import find_namespace_packages, setup  # type: ignore

metadata: dict = {}
with open("_metadata.py") as fp:
    exec(fp.read(), metadata)
shutil.copy(
    "_metadata.py",
    os.path.join("pytket", "extensions", "azure", "_metadata.py"),
)

long_description = """
[Pytket](https://tket.quantinuum.com/api-docs/index.html) is a python module
providing an extensive set of tools for compiling and executing quantum
circuits.

The `pytket-azure` extension allows `pytket` circuits to be submitted and
executed on various devices and simulators via
[Azure Quantum](https://learn.microsoft.com/en-us/azure/quantum/).

Note that this package is still at an early, experimental stage of development.
"""

setup(
    name="pytket-azure",
    version=metadata["__extension_version__"],
    author="TKET development team",
    author_email="tket-support@quantinuum.com",
    python_requires=">=3.10",
    project_urls={
        "Documentation": "https://tket.quantinuum.com/extensions/pytket-azure/index.html",
        "Source": "https://github.com/CQCL/pytket-azure",
        "Tracker": "https://github.com/CQCL/pytket-azure/issues",
    },
    description="Extension for pytket, providing access to Azure devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="Apache 2",
    packages=find_namespace_packages(include=["pytket.*"]),
    include_package_data=True,
    install_requires=[
        "azure-quantum >= 2.2.0",
        "pytket >= 1.37.0",
        "pytket-qir >= 0.19.0",
    ],
    classifiers=[
        "Environment :: Console",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    zip_safe=False,
)
