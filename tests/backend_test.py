# Copyright 2019-2024 Quantinuum
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

from collections import Counter
import os
from warnings import warn
import pytest
from pytket.circuit import Circuit
from pytket.extensions.azure import AzureBackend

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
REASON = "PYTKET_RUN_REMOTE_TESTS not set (requires Azure credentials)"


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["ionq.simulator"], indirect=True)
def test_ionq_simulator(azure_backend: AzureBackend) -> None:
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    b = azure_backend
    c1 = b.get_compiled_circuit(c)
    if b.is_available() and b.average_queue_time_s() < 60:
        h = b.process_circuit(c1, n_shots=10)
        r = b.get_result(h)
        counts = r.get_counts()
        assert counts == Counter({(0, 0): 5, (1, 1): 5})
    else:
        warn("ionq.simulator unavailable or queue time >= 60s: not submitting")


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11e(azure_backend: AzureBackend) -> None:
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    b = azure_backend
    c1 = b.get_compiled_circuit(c)
    if b.is_available() and b.average_queue_time_s() < 60:
        h = b.process_circuit(c1, n_shots=1000)
        r = b.get_result(h)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        warn("quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting")


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1e"], indirect=True)
def test_quantinuum_option_params(azure_backend: AzureBackend) -> None:
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    b = azure_backend
    c1 = b.get_compiled_circuit(c)
    if b.is_available() and b.average_queue_time_s() < 600:
        h = b.process_circuit(c1, n_shots=1000, option_params={"error_model": False})
        r = b.get_result(h)
        counts = r.get_counts()
        assert all(x0 == x1 for x0, x1 in counts.keys())
    else:
        warn("quantinuum.sim.h1-1e unavailable or queue time >= 600s: not submitting")
