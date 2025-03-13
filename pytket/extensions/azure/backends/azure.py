# Copyright Quantinuum
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
import warnings
from ast import literal_eval
from collections import Counter
from collections.abc import Sequence
from enum import Enum
from functools import cache
from typing import Any, Optional, Union, cast

from azure.quantum import Job, Workspace
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Circuit, OpType
from pytket.extensions.azure._metadata import __extension_version__
from pytket.passes import (
    AutoRebase,
    AutoSquash,
    BasePass,
    DecomposeBoxes,
    DecomposeTK2,
    FlattenRelabelRegistersPass,
    FullPeepholeOptimise,
    GreedyPauliSimp,
    NormaliseTK2,
    RemoveBarriers,
    RemovePhaseOps,
    RemoveRedundancies,
    SequencePass,
    SynthesiseTK,
    ZZPhaseToRz,
    scratch_reg_resize_pass,
)
from pytket.predicates import (
    GateSetPredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.qir import QIRFormat, QIRProfile, pytket_to_qir
from pytket.utils import OutcomeArray

from .config import AzureConfig


class DeviceType(Enum):
    """Different types of devices"""

    Quantinuum = 0
    Ionq = 1
    Rigetti = 2
    Default = 3


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
    OpType.CY,
    OpType.CZ,
    OpType.H,
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
    OpType.ZZPhase,
}


_ADDITIONAL_GATES = {
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
    OpType.RangePredicate,
    OpType.MultiBit,
    OpType.ExplicitPredicate,
    OpType.ExplicitModifier,
    OpType.SetBits,
    OpType.CopyBits,
    OpType.ClExpr,
}


_QUANTINUUM_TARGET_GATESET = {
    OpType.ZZPhase,  # 2 qubit gate
    OpType.Rz,
    OpType.Rx,
}

_ALL_GATES = _ADDITIONAL_GATES.copy()
_ALL_GATES.update(_GATE_SET)


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
        self._result_bits: dict[ResultHandle, list] = {}
        self._result_c_regs: dict[ResultHandle, list] = {}

        self._device_type = DeviceType.Default

        if self._backendinfo.device_name:
            if self._backendinfo.device_name[:11] == "quantinuum.":
                self._device_type = DeviceType.Quantinuum
            elif self._backendinfo.device_name[:5] == "ionq.":
                self._device_type = DeviceType.Ionq
            elif self._backendinfo.device_name[:8] == "rigetti.":
                self._device_type = DeviceType.Rigetti
            else:
                warnings.warn(
                    f"Unknown device type for {self._backendinfo.device_name},\
using default compilation"
                )
        else:
            warnings.warn("Unknown device type, using default compilation")

    @property
    def backend_info(self) -> BackendInfo:
        return self._backendinfo

    @property
    def required_predicates(self) -> list[Predicate]:
        return [GateSetPredicate(_ALL_GATES), NoSymbolsPredicate()]

    def _default_2q_gate(self, device_name: str) -> OpType:
        return OpType.ZZPhase

    def rebase_pass(self) -> BasePass:
        if self._device_type == DeviceType.Quantinuum:
            return AutoRebase(
                _QUANTINUUM_TARGET_GATESET,
                allow_swaps=True,
            )
        else:
            return AutoRebase(gateset=_GATE_SET)

    def default_compilation_pass(
        self, optimisation_level: int = 2, timeout: int = 300
    ) -> BasePass:
        """
        :param optimisation_level: Allows values of 0, 1, 2 or 3, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :param timeout: Only valid for optimisation level 3, gives a maximimum time
            for running a single thread of the pass `GreedyPauliSimp`. Increase for
            optimising larger circuits.

        :return: Compilation pass for compiling circuits to Quantinuum devices
        """
        assert optimisation_level in range(4)

        if self._device_type != DeviceType.Quantinuum:
            return self.rebase_pass()

        passlist = [
            DecomposeBoxes(),
            scratch_reg_resize_pass(),
        ]
        squash = AutoSquash({OpType.Rx, OpType.Rz})
        target_2qb_gate = OpType.ZZPhase
        assert target_2qb_gate is not None
        decomposition_passes = [
            NormaliseTK2(),
            DecomposeTK2(
                allow_swaps=True,
                ZZPhase_fidelity=1.0,
            ),
        ]

        if optimisation_level == 0:
            passlist.append(self.rebase_pass())
        elif optimisation_level == 1:
            passlist.append(SynthesiseTK())
            passlist.extend(decomposition_passes)
            passlist.extend(
                [
                    self.rebase_pass(),
                    ZZPhaseToRz(),
                    RemoveRedundancies(),
                    squash,
                    RemoveRedundancies(),
                ]
            )
        elif optimisation_level == 2:
            passlist.append(
                FullPeepholeOptimise(
                    allow_swaps=True,
                    target_2qb_gate=OpType.TK2,
                )
            )
            passlist.extend(decomposition_passes)
            passlist.extend(
                [
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    RemoveRedundancies(),
                ]
            )
        else:
            passlist.extend(
                [
                    RemoveBarriers(),
                    AutoRebase(
                        {
                            OpType.Z,
                            OpType.X,
                            OpType.Y,
                            OpType.S,
                            OpType.Sdg,
                            OpType.V,
                            OpType.Vdg,
                            OpType.H,
                            OpType.CX,
                            OpType.CY,
                            OpType.CZ,
                            OpType.SWAP,
                            OpType.Rz,
                            OpType.Rx,
                            OpType.Ry,
                            OpType.T,
                            OpType.Tdg,
                            OpType.ZZMax,
                            OpType.ZZPhase,
                            OpType.XXPhase,
                            OpType.YYPhase,
                        }
                    ),
                    GreedyPauliSimp(
                        allow_zzphase=True,
                        only_reduce=True,
                        thread_timeout=timeout,
                        trials=10,
                    ),
                ]
            )
            passlist.extend(decomposition_passes)
            passlist.extend(
                [
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    RemoveRedundancies(),
                ]
            )
        passlist.append(RemovePhaseOps())

        passlist.append(FlattenRelabelRegistersPass("q"))
        return SequencePass(passlist)

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

            input_params = {
                "entryPoint": "main",
                "arguments": [],
                "count": n_shots,
            }

            if self._device_type == DeviceType.Quantinuum:

                module_bitcode = pytket_to_qir(
                    c,
                    qir_format=QIRFormat.STRING,
                    int_type=64,
                    cut_pytket_register=False,
                    profile=QIRProfile.AZUREADAPTIVE,
                )
            else:
                module_bitcode = pytket_to_qir(
                    c,
                    qir_format=QIRFormat.STRING,
                    int_type=64,
                    cut_pytket_register=False,
                    profile=QIRProfile.AZUREBASE,
                )

            if option_params is not None:
                input_params.update(option_params)  # type: ignore
            job = self._target.submit(
                input_data=module_bitcode,
                input_data_format="qir.v1",
                output_data_format="microsoft.quantum-results.v1",
                name=f"job_{i}",
                input_params=input_params,
            )
            jobid: str = job.id
            handle = ResultHandle(jobid)
            handles.append(handle)
            self._jobs[handle] = job
            self._result_bits[handle] = c.bits
            self._result_c_regs[handle] = c.c_registers
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

    def _make_backend_result(
        self, results: Any, job: Job, handle: ResultHandle
    ) -> BackendResult:
        n_shots = job.details.input_params["count"]
        counts: Counter[OutcomeArray] = Counter()
        if self._device_type == DeviceType.Quantinuum:
            for s, p in results.items():
                outcome = literal_eval(s)
                n = int(n_shots * p + 0.5)
                assert len(outcome) == len(self._result_c_regs[handle])
                list_bits: list = []
                for res, creg in zip(outcome, self._result_c_regs[handle]):
                    long_res = bin(int(res)).replace(
                        "0b",
                        "0000000000000000000000000000000000000\
00000000000000000000000000",  # 0 * 63
                    )
                    list_bits.append(long_res[-1 : -creg.size - 1 : -1])

                all_bits = "".join(list_bits)

                counts[OutcomeArray.from_readouts([[int(x) for x in all_bits]])] = n
            return BackendResult(counts=counts, c_bits=self._result_bits[handle])
        else:
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
                {"result": self._make_backend_result(results, job, handle)},
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
                raise RuntimeError(f"Circuit has errored. {circuit_status}")

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
