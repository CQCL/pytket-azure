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

"""Azure config."""

from dataclasses import dataclass
from typing import Any, ClassVar, Optional

from pytket.config import PytketExtConfig


@dataclass
class AzureConfig(PytketExtConfig):
    """Holds config parameters for pytket-azure."""

    ext_dict_key: ClassVar[str] = "azure"

    resource_id: Optional[str]
    location: Optional[str]
    connection_string: Optional[str]
    use_string: bool = False

    @classmethod
    def from_extension_dict(
        cls: type["AzureConfig"], ext_dict: dict[str, Any]
    ) -> "AzureConfig":
        return cls(
            ext_dict.get("resource_id"),
            ext_dict.get("location"),
            ext_dict.get("connection_string"),
            ext_dict.get("use_string"),
        )


def set_azure_config(
    resource_id: Optional[str] = None,
    location: Optional[str] = None,
    connection_string: Optional[str] = None,
    use_string: bool = False,
) -> None:
    """Save Azure confuguration."""
    config = AzureConfig.from_default_config_file()
    if use_string:
        config.use_string = use_string
        if connection_string is not None:
            config.connection_string = connection_string
    else:
        if resource_id is not None:
            config.resource_id = resource_id
        if location is not None:
            config.location = location
    config.update_default_config_file()
