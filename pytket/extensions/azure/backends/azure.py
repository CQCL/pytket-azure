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
from ast import literal_eval
from collections import Counter
from collections.abc import Sequence
from functools import cache
from typing import Any, Optional, Union, cast

from qiskit_qir import to_qir_module

from azure.quantum import Job, Workspace
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType
from pytket.extensions.azure._metadata import __extension_version__
from pytket.extensions.qiskit import tk_to_qiskit
from pytket.passes import AutoRebase, BasePass
from pytket.predicates import GateSetPredicate, Predicate
from pytket.utils import OutcomeArray

from .config import AzureConfig


def _get_workspace(
    resource_id: Optional[str] = None,
    location: Optional[str] = None,
    connection_string: Optional[str] = None,
) -> Workspace:
    if os.getenv("AZURE_QUANTUM_CONNECTION_STRING") is not None:
        return Workspace()
    else:
        config = AzureConfig.from_default_config_file()
        if config.use_string:
            if connection_string is None:
                connection_string = config.connection_string
            return Workspace.from_connection_string(connection_string)
        else:
            if resource_id is None:
                resource_id = config.resource_id
            if location is None:
                location = config.location
            return Workspace(resource_id=resource_id, location=location)


_GATE_SET = {
    OpType.CX,
    OpType.CZ,
    OpType.H,
    OpType.Measure,
    OpType.Rx,
    OpType.Ry,
    OpType.Rz,
    OpType.S,
    OpType.Sdg,
    OpType.SWAP,
    OpType.T,
    OpType.Tdg,
    OpType.X,
    OpType.Y,
    OpType.Z,
}


class AzureBackend(Backend):
    """Interface to Azure Quantum."""

    def __init__(
        self,
        name: str,
        resource_id: Optional[str] = None,
        location: Optional[str] = None,
        connection_string: Optional[str] = None,
        use_string: bool = False,
    ):
        """Construct an Azure backend for a device.

        If the environment variable `AZURE_QUANTUM_CONNECTION_STRING` is set,
        this is used for authentication. Otherwise, the Azure Quantum
        `resource_id` and `location` are read from pytket config, if set, or
        else from the provided arguments.


        :param name: Device name. Use `AzureBackend.available_devices()` to
            obtain a list of possible device names.
        :param resource_id: Azure Quantum `resource_id`. If omitted this is read
            from config (see `set_azure_config()`), unless the environment
            variable `AZURE_QUANTUM_CONNECTION_STRING` is set in which case this
            is used.
        :param location: Azure Quantum `location`. If omitted this is read from
            config (see `set_azure_config()`), unless the environment variable
            `AZURE_QUANTUM_CONNECTION_STRING` is set in which case this is used.
        :param connection_string: Azure Quantum `connection_string`.
            The connection_string can be set on Azure Quantum.
            See https://learn.microsoft.com/en-us/azure/quantum/how-to-connect-workspace
            If omitted this is read from config (see `set_azure_config()`), unless
            the environment variable `AZURE_QUANTUM_CONNECTION_STRING` is set in which
            case this is used.
        :param use_string: Use the `connection_string`. Defaults to False.
        """
        super().__init__()
        if use_string:
            self._workspace = _get_workspace(connection_string=connection_string)
        else:
            self._workspace = _get_workspace(resource_id, location)
        self._target = self._workspace.get_targets(name=name)
        self._backendinfo = BackendInfo(
            name=type(self).__name__,
            device_name=name,
            version=__extension_version__,
            architecture=None,
            gate_set=_GATE_SET,
        )
        _persistent_handles = False
        self._jobs: dict[ResultHandle, Job] = {}

    @property
    def backend_info(self) -> BackendInfo:
        return self._backendinfo

    @property
    def required_predicates(self) -> list[Predicate]:
        return [GateSetPredicate(_GATE_SET)]

    def rebase_pass(self) -> BasePass:
        return AutoRebase(gateset=_GATE_SET)

    def default_compilation_pass(self, optimisation_level: int = 1) -> BasePass:
        return self.rebase_pass()

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return (str,)

    def process_circuits(
        self,
        circuits: Sequence[Circuit],
        n_shots: Union[None, int, Sequence[Optional[int]]] = None,
        valid_check: bool = True,
        **kwargs: KwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.Backend.process_circuits`.

        Supported kwargs:

        - option_params: a dictionary with string keys and arbitrary values;
          key-value pairs in the dictionary are passed as input parameters to
          the backend. Their semantics are backend-dependent.
        """
        option_params = kwargs.get("option_params")
        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(
            n_shots,
            len(circuits),
            optional=False,
        )

        if valid_check:
            self._check_all_circuits(circuits)

        handles = []
        for i, (c, n_shots) in enumerate(zip(circuits, n_shots_list)):
            qkc = tk_to_qiskit(c)
            module, entry_points = to_qir_module(qkc)
            assert len(entry_points) == 1
            input_params = {
                "entryPoint": entry_points[0],
                "arguments": [],
                "count": n_shots,
            }
            if option_params is not None:
                input_params.update(option_params)  # type: ignore
            job = self._target.submit(
                input_data=module.bitcode,
                input_data_format="qir.v1",
                output_data_format="microsoft.quantum-results.v1",
                name=f"job_{i}",
                input_params=input_params,
            )
            jobid: str = job.id
            handle = ResultHandle(jobid)
            handles.append(handle)
            self._jobs[handle] = job
        for handle in handles:
            self._cache[handle] = dict()
        return handles

    def _update_cache_result(
        self, handle: ResultHandle, result_dict: dict[str, BackendResult]
    ) -> None:
        if handle in self._cache:
            self._cache[handle].update(result_dict)
        else:
            self._cache[handle] = result_dict

    def _make_backend_result(self, results: Any, job: Job) -> BackendResult:
        n_shots = job.details.input_params["count"]
        counts: Counter[OutcomeArray] = Counter()
        for s, p in results.items():
            outcome = literal_eval(s)
            n = int(n_shots * p + 0.5)
            oa = OutcomeArray.from_readouts([outcome])
            counts[oa] = n
        return BackendResult(counts=counts)

    def circuit_status(self, handle) -> CircuitStatus:
        job = self._jobs[handle]
        job.refresh()
        status = job.details.status
        if status == "Succeeded":
            results = job.get_results()
            self._update_cache_result(
                handle,
                {"result": self._make_backend_result(results, job)},
            )
            return CircuitStatus(StatusEnum.COMPLETED)
        elif status == "Waiting":
            return CircuitStatus(StatusEnum.QUEUED)
        elif status == "Executing":
            return CircuitStatus(StatusEnum.RUNNING)
        elif status == "Failed":
            return CircuitStatus(StatusEnum.ERROR, job.details.error_data.message)
        else:
            return CircuitStatus(
                StatusEnum.ERROR, f"Unrecognized job status: '{status}'"
            )

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.

        Supported kwargs:

        - timeout (int): timeout in seconds
        """
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            self._jobs[handle].wait_until_completed(timeout_secs=kwargs.get("timeout"))
            circuit_status = self.circuit_status(handle)
            if circuit_status.status is StatusEnum.COMPLETED:
                return cast(BackendResult, self._cache[handle]["result"])
            else:
                assert circuit_status.status is StatusEnum.ERROR
                raise RuntimeError("Circuit has errored.")

    def is_available(self) -> bool:
        """Availability reported by the target."""
        self._target.refresh()
        return self._target.current_availability == "Available"

    def average_queue_time_s(self) -> int:
        """Average queue time in seconds reported by the target."""
        self._target.refresh()
        return self._target.average_queue_time

    @classmethod
    @cache
    def available_devices(cls, **kwargs: Any) -> list[BackendInfo]:
        """
        See :py:meth:`pytket.backends.Backend.get_result`.

        Supported kwargs:

        - resource_id (str)
        - location (str)
        - connection_string (str)
        - use_string (bool) = False

        If omitted these are read from config, unless the environment variable
        `AZURE_QUANTUM_CONNECTION_STRING` is set in which case it is used.
        """
        if kwargs.get("use_string"):
            connection_string = kwargs.get("connection_string")
            workspace = _get_workspace(connection_string=connection_string)
        else:
            resource_id = kwargs.get("resource_id")
            location = kwargs.get("location")
            workspace = _get_workspace(resource_id, location)
        return [
            BackendInfo(
                name=cls.__name__,
                device_name=target.name,
                version=__extension_version__,
                architecture=None,
                gate_set=_GATE_SET,
            )
            for target in workspace.get_targets()
        ]
