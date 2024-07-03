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

import pytest
from _pytest.fixtures import SubRequest

from pytket.extensions.azure import AzureBackend


@pytest.fixture(name="azure_backend")
def fixture_azure_backend(request: SubRequest) -> AzureBackend:
    return AzureBackend(
        name=request.param,
        resource_id=os.getenv("PYTKET_REMOTE_AZURE_RESOURCE_ID"),
        location=os.getenv("PYTKET_REMOTE_AZURE_LOCATION"),
    )
