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
from collections import Counter

import pytest

from pytket.circuit import CircBox, Circuit, Qubit, if_not_bit
from pytket.circuit.clexpr import ClExpr, ClOp, ClRegVar, WiredClExpr
from pytket.circuit.logic_exp import (
    reg_eq,
    reg_geq,
    reg_gt,
    reg_leq,
    reg_lt,
    reg_neq,
)
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
        r = b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert counts == Counter({(0, 0): 5, (1, 1): 5})
    else:
        raise ValueError(
            "ionq.simulator unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc(azure_backend: AzureBackend) -> None:
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    b = azure_backend
    c1 = b.get_compiled_circuit(c)
    if b.is_available() and b.average_queue_time_s() < 60:
        h = b.process_circuit(c1, n_shots=1000)
        r = b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc_complex_circuit(azure_backend: AzureBackend) -> None:
    circ = Circuit(4, 4)
    circ.H(0)
    circ.X(0)
    circ.Y(0)
    circ.Z(0)
    circ.Rx(0.5, 0)
    circ.ZZPhase(0.5, 0, 1)
    circ.PhasedX(0.5, 0.4, 1)
    circ.ZZMax(0, 1)

    circ.CX(1, 2)
    circ.CX(1, 3)
    circ.H(1)
    circ.Measure(1, 1)
    b = azure_backend
    c1 = b.get_compiled_circuit(circ)
    if b.is_available() and b.average_queue_time_s() < 60:
        h = b.process_circuit(c1, n_shots=1000)
        r = b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc_complex_circuit_2(azure_backend: AzureBackend) -> None:
    c = Circuit(4)
    c.X(0)
    c.Y(1)
    c.Z(2)
    c.H(3)
    cbox = CircBox(c)
    d = Circuit(4)
    a = d.add_c_register("a", 4)
    d.add_circbox(cbox, [0, 2, 1, 3], condition=a[0])
    d.Measure(Qubit(0), a[0])

    b = azure_backend
    c1 = b.get_compiled_circuit(d)
    if b.is_available() and b.average_queue_time_s() < 60:
        h = b.process_circuit(c1, n_shots=1000)
        r = b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc_complex_circuit_3(azure_backend: AzureBackend) -> None:
    circ = Circuit(4, 4)
    circ.H(0)
    circ.X(0)
    circ.Y(0)
    circ.Z(0)
    circ.Rx(0.5, 0)
    circ.Ry(0.5, 0)
    circ.Rz(0.5, 0)
    circ.ZZPhase(0.5, 0, 1)
    circ.PhasedX(0.5, 0.4, 1)
    circ.ZZMax(0, 1)
    circ.T(1)
    circ.CX(2, 3)
    circ.CY(2, 3)
    circ.CZ(2, 3)
    circ.SWAP(2, 3)

    circ.CX(1, 2)
    circ.CX(1, 3)
    circ.H(1)
    circ.Measure(1, 1)
    b = azure_backend
    c1 = b.get_compiled_circuit(circ, 0)
    if b.is_available() and b.average_queue_time_s() < 60:
        h = b.process_circuit(c1, n_shots=1000)
        r = b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc_two_regs(azure_backend: AzureBackend) -> None:
    c = Circuit(2, name="test_classical")
    a = c.add_c_register("a", 10)
    b = c.add_c_register("b", 11)

    c.Measure(Qubit(0), a[0])
    c.Measure(Qubit(1), b[0])

    a_b = azure_backend
    c1 = a_b.get_compiled_circuit(c)
    if a_b.is_available() and a_b.average_queue_time_s() < 60:
        h = a_b.process_circuit(c1, n_shots=1000)
        r = a_b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc_reset_gate(azure_backend: AzureBackend) -> None:
    c = Circuit(2, name="test_classical")
    a = c.add_c_register("a", 10)
    b = c.add_c_register("b", 11)

    c.Measure(Qubit(0), a[0])
    c.Measure(Qubit(1), b[0])
    c.Reset(Qubit(0))
    c.H(Qubit(0))
    c.Measure(Qubit(0), a[1])

    a_b = azure_backend
    c1 = a_b.get_compiled_circuit(c)
    if a_b.is_available() and a_b.average_queue_time_s() < 60:
        h = a_b.process_circuit(c1, n_shots=1000)
        r = a_b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11sc_complex(azure_backend: AzureBackend) -> None:
    c = Circuit(1, name="test_classical")
    a = c.add_c_register("a", 10)
    b = c.add_c_register("b", 11)
    d = c.add_c_register("d", 20)

    c.Measure(Qubit(0), a[0])

    c.add_c_setbits([True, True] + [False] * 9, list(b))

    c.add_clexpr(
        expr=WiredClExpr(
            expr=ClExpr(op=ClOp.RegAdd, args=[ClRegVar(0), ClRegVar(1)]),
            reg_posn={0: list(range(10)), 1: list(range(10, 21))},
            output_posn=list(range(21, 41)),
        ),
        args=a.to_list() + b.to_list() + d.to_list(),
    )
    a_b = azure_backend
    c1 = a_b.get_compiled_circuit(c)
    if a_b.is_available() and a_b.average_queue_time_s() < 60:
        h = a_b.process_circuit(c1, n_shots=1000)
        r = a_b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11e_cond(azure_backend: AzureBackend) -> None:
    c = Circuit(1, name="test_classical")
    a = c.add_c_register("a", 32)
    b = c.add_c_register("b", 32)
    d = c.add_c_register("d", 32)

    c.Measure(Qubit(0), a[0])

    c.add_c_setreg(23, b)

    c.add_clexpr(
        expr=WiredClExpr(
            expr=ClExpr(op=ClOp.RegAdd, args=[ClRegVar(0), ClRegVar(1)]),
            reg_posn={0: list(range(32)), 1: list(range(32, 64))},
            output_posn=list(range(64, 96)),
        ),
        args=a.to_list() + b.to_list() + d.to_list(),
    )

    c.X(0, condition=a[0])
    c.Measure(Qubit(0), b[4])

    a_b = azure_backend
    c1 = a_b.get_compiled_circuit(c)
    if a_b.is_available() and a_b.average_queue_time_s() < 60:
        h = a_b.process_circuit(c1, n_shots=1000)
        r = a_b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1sc"], indirect=True)
def test_quantinuum_sim_h11e_cond_2(azure_backend: AzureBackend) -> None:
    c = Circuit(1, name="test_classical")
    a = c.add_c_register("a", 32)
    b = c.add_c_register("b", 32)
    d = c.add_c_register("d", 32)

    c.Measure(Qubit(0), a[0])

    c.add_c_setreg(23, b)

    c.add_clexpr(
        expr=WiredClExpr(
            expr=ClExpr(op=ClOp.RegAdd, args=[ClRegVar(0), ClRegVar(1)]),
            reg_posn={0: list(range(32)), 1: list(range(32, 64))},
            output_posn=list(range(64, 96)),
        ),
        args=a.to_list() + b.to_list() + d.to_list(),
    )
    c.add_clexpr(
        expr=WiredClExpr(
            expr=ClExpr(op=ClOp.RegSub, args=[ClRegVar(0), ClRegVar(1)]),
            reg_posn={0: list(range(32)), 1: list(range(32, 64))},
            output_posn=list(range(64, 96)),
        ),
        args=a.to_list() + b.to_list() + d.to_list(),
    )
    c.add_clexpr(
        expr=WiredClExpr(
            expr=ClExpr(op=ClOp.RegLsh, args=[ClRegVar(0), 1]),
            reg_posn={0: list(range(32))},
            output_posn=list(range(32)),
        ),
        args=a.to_list(),
    )
    c.add_clexpr(
        expr=WiredClExpr(
            expr=ClExpr(op=ClOp.RegRsh, args=[ClRegVar(0), 1]),
            reg_posn={0: list(range(32))},
            output_posn=list(range(32, 64)),
        ),
        args=a.to_list() + b.to_list(),
    )

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.X(0, condition=(a[0] ^ b[0]))
    c.X(0, condition=reg_eq(a & b, 1))
    c.X(0, condition=reg_eq(a | b, 1))

    c.X(0, condition=a[0])
    c.Measure(Qubit(0), b[4])

    c.X(0, condition=reg_neq(a, 1))
    c.X(0, condition=if_not_bit(a[0]))
    c.X(0, condition=reg_gt(a, 1))
    c.X(0, condition=reg_lt(a, 1))
    c.X(0, condition=reg_geq(a, 1))
    c.X(0, condition=reg_leq(a, 1))
    c.Measure(Qubit(0), b[4])
    a_b = azure_backend
    c1 = a_b.get_compiled_circuit(c)
    if a_b.is_available() and a_b.average_queue_time_s() < 60:
        h = a_b.process_circuit(c1, n_shots=1000)
        r = a_b.get_result(h, timeout=120)
        counts = r.get_counts()
        assert sum(counts.values()) == 1000
    else:
        raise ValueError(
            "quantinuum.sim.h1-1sc unavailable or queue time >= 60s: not submitting"
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("azure_backend", ["quantinuum.sim.h1-1e"], indirect=True)
def test_quantinuum_option_params(azure_backend: AzureBackend) -> None:
    c = Circuit(2, 2).H(0).CX(0, 1).measure_all()
    a_b = azure_backend
    c1 = a_b.get_compiled_circuit(c)
    if a_b.is_available() and a_b.average_queue_time_s() < 600:
        h = a_b.process_circuit(c1, n_shots=1000, option_params={"error_model": False})  # type: ignore
        r = a_b.get_result(h, timeout=1200)
        counts = r.get_counts()
        assert all(x[0] == x[1] for x in counts)
        assert any(x[0] == 1 for x in counts)  # might fail in very rare cases
    else:
        raise ValueError(
            "quantinuum.sim.h1-1e unavailable or queue time >= 600s: not submitting"
        )
