"""
Hidden self-check tests for the student implementations in Part 4 of the
CVRP notebook.

These are *not* part of the assignment. They give students fast, localized
feedback on the two classes they implement:

    - GROUP_STATE        (Part 4.1)  -> test_group_state(GROUP_STATE, DEVICE)
    - GROUP_ENVIRONMENT  (Part 4.2)  -> test_group_environment(GROUP_ENVIRONMENT, DEVICE)

Design goals:
    - Each check prints "✅ <name>" or "❌ <name> — Hint: ...".
    - A failure (wrong value) or an exception (e.g. a blank left as `...`)
      never raises out of the test: it is caught and turned into a hint, so a
      single mistake does not crash the rest of the notebook.
    - The student's class is passed in explicitly (it is defined in a notebook
      cell and monkey-patched onto `cvrp`), together with the active DEVICE so
      fixtures live on the same device as the tensors the student creates.
"""

import numpy as np
import torch

# Small tolerance for floating-point demand/cost arithmetic.
_RTOL = 1e-5
_ATOL = 1e-5


# --------------------------------------------------------------------------- #
# Tiny reporting / running helpers
# --------------------------------------------------------------------------- #
def _check(name, passed, hint=""):
    """Print a single ✅/❌ line and return whether it passed."""
    if passed:
        print(f"  ✅ {name}")
        return True
    line = f"  ❌ {name}"
    if hint:
        line += f" — Hint: {hint}"
    print(line)
    return False


def _safe(name, hint, fn):
    """Run a check `fn` returning a bool; turn any exception into a hint.

    This is what makes a not-yet-filled blank (`...`) or a shape mismatch show
    up as a friendly ❌ instead of a raw traceback that stops the notebook.
    """
    try:
        passed = bool(fn())
    except Exception as exc:  # noqa: BLE001 - we intentionally catch everything
        print(f"  ❌ {name} — Hint: {hint} (raised {type(exc).__name__}: {exc})")
        return False
    return _check(name, passed, hint)


def _summary(title, results):
    """Print a header before, and a pass/fail tally after, a group of checks."""
    n_pass = sum(results)
    n_total = len(results)
    if n_pass == n_total:
        print(f"\n🎉 {title}: all {n_total} checks passed!\n")
    else:
        print(f"\n⚠️  {title}: {n_pass}/{n_total} checks passed — see hints above.\n")
    return n_pass == n_total


def _allclose(a, b):
    return torch.allclose(a, b, rtol=_RTOL, atol=_ATOL)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
# Vehicle capacity used when building GROUP_STATE fixtures. Demands are small
# integers in [0, _CAP]; passing the capacity explicitly keeps the checks
# independent of `source.parameters.VEHICLE_CAPACITY`.
_CAP = 10


def _make_data(n_nodes, batch=1, n_feature_cols=3, device="cpu"):
    """Build a placeholder feature tensor of shape (batch, problem+1, cols).

    GROUP_STATE reads only `data.size(0)` and `data.size(1)` from this tensor
    (demands now come from `int_demand`, not from `data`), so the contents are
    irrelevant to the methods under test and are left as zeros.
    """
    return torch.zeros((batch, n_nodes, n_feature_cols),
                       dtype=torch.float32, device=device)


def _make_int_demand(demands, device="cpu"):
    """Build an integer demand tensor of shape (batch, problem+1, 1).

    `demands` is a list-of-lists: one row of integer node demands per batch item
    (entry 0 is the depot and should be 0).
    """
    d = torch.as_tensor(demands, dtype=torch.long, device=device)
    return d[:, :, None]


class _StubState:
    """Stand-in for GROUP_STATE used to isolate GROUP_ENVIRONMENT methods.

    It exposes just enough surface for `_get_travel_cost` and `step`:
      - `selected_node_list`: the visited-node tours,
      - `finished`: the per-tour finished flags,
      - `move_to`: a recording no-op so `step` can be tested in isolation.
    """

    def __init__(self, selected_node_list, finished=None):
        self.selected_node_list = selected_node_list
        self.finished = finished
        self.move_to_called = False
        self.move_to_arg = None

    def move_to(self, selected_idx_mat):
        self.move_to_called = True
        self.move_to_arg = selected_idx_mat


# The cost fixture shared by _get_travel_cost and step:
# 2 problem instances, each with its own 4x4 cost matrix, 3 tours per instance,
# tours of length 6 (with trailing zeros acting as padding / depot returns).
_COST_MATRIX = torch.tensor([
    [[0.0, 11.0, 5.0, 30.0],
     [7.0, 0.0, 4.0, 25.0],
     [13.0, 15.0, 0.0, 35.0],
     [33.0, 21.0, 33.0, 0.0]],
    [[0.0, 3.0, 11.0, 5.0],
     [4.0, 0.0, 13.0, 12.0],
     [8.0, 11.0, 0.0, 24.0],
     [7.0, 8.0, 20.0, 0.0]],
])
_SELECTED_NODES = torch.tensor([
    [[0, 1, 2, 3, 0, 0],
     [0, 2, 0, 1, 3, 0],
     [0, 3, 2, 1, 0, 0]],
    [[0, 1, 0, 3, 2, 0],
     [0, 2, 0, 1, 3, 0],
     [0, 3, 1, 0, 2, 0]],
], dtype=torch.long)
_EXPECTED_COSTS = torch.tensor([[83.0, 87.0, 85.0],
                                [40.0, 41.0, 36.0]])


# --------------------------------------------------------------------------- #
# Part 4.1 — GROUP_STATE
# --------------------------------------------------------------------------- #
def test_group_state(GROUP_STATE, DEVICE):
    """Check GROUP_STATE.__init__ and GROUP_STATE.move_to."""
    print("Running tests for GROUP_STATE (Part 4.1)...")
    results = []

    # ---- __init__ -------------------------------------------------------- #
    # Check all four state tensors across several (batch, n_nodes, group)
    # settings so their shapes must genuinely derive from data.size(0),
    # data.size(1) and group_size — `problem+1` equals n_nodes here.
    state = None
    for batch_s, n_nodes, group_s in [(2, 4, 3), (1, 6, 4), (3, 3, 2)]:
        data = _make_data(n_nodes, batch=batch_s, device=DEVICE)
        int_demand = torch.zeros((batch_s, n_nodes, 1), dtype=torch.long, device=DEVICE)
        state = GROUP_STATE(group_size=group_s, data=data, cost_matrix =_COST_MATRIX.to(DEVICE),
                            int_demand=int_demand, vehicle_capacity=_CAP)
        s2 = (batch_s, group_s)                 # (batch, group)
        s3 = (batch_s, group_s, n_nodes)        # (batch, group, problem+1)

        results.append(_safe(
            f"__init__: remaining_capacity is full ({_CAP}) of shape {s2}",
            "self.remaining_capacity should be a LongTensor filled with "
            "vehicle_capacity, with shape (batch_s, group_s).",
            lambda st=state, sh=s2: tuple(st.remaining_capacity.shape) == sh
            and st.remaining_capacity.dtype == torch.long
            and torch.equal(st.remaining_capacity,
                            torch.full(sh, _CAP, dtype=torch.long, device=DEVICE)),
        ))
        results.append(_safe(
            f"__init__: visited_ninf_flag is zeros of shape {s3}",
            "self.visited_ninf_flag should be a FloatTensor of zeros with shape "
            "(batch_s, group_s, problem_size+1).",
            lambda st=state, sh=s3: tuple(st.visited_ninf_flag.shape) == sh
            and st.visited_ninf_flag.dtype == torch.float32
            and _allclose(st.visited_ninf_flag, torch.zeros(sh, device=DEVICE)),
        ))
        results.append(_safe(
            f"__init__: ninf_mask is zeros of shape {s3}",
            "self.ninf_mask should be a FloatTensor of zeros with shape "
            "(batch_s, group_s, problem_size+1).",
            lambda st=state, sh=s3: tuple(st.ninf_mask.shape) == sh
            and st.ninf_mask.dtype == torch.float32
            and _allclose(st.ninf_mask, torch.zeros(sh, device=DEVICE)),
        ))
        results.append(_safe(
            f"__init__: finished is all-False BoolTensor of shape {s2}",
            "self.finished should be a BoolTensor of zeros (False) with shape "
            "(batch_s, group_s).",
            lambda st=state, sh=s2: tuple(st.finished.shape) == sh
            and st.finished.dtype == torch.bool
            and not st.finished.any(),
        ))

    # On CPU this is a no-op, but a missing `.to(DEVICE)` causes device-mismatch
    # errors once training runs on GPU/MPS — so check placement explicitly.
    expected_dev = torch.device(DEVICE).type
    results.append(_safe(
        "__init__: tensors are placed on DEVICE",
        "Create each tensor on the right device with `.to(DEVICE)`.",
        lambda: all(
            t.device.type == expected_dev
            for t in (state.remaining_capacity, state.visited_ninf_flag,
                      state.ninf_mask, state.finished)
        ),
    ))

    # ---- move_to --------------------------------------------------------- #
    # A single move_to from a fresh state is checked across several scenarios so
    # the behaviour is exercised genuinely rather than for one lucky fixture.
    # Depot positions vary (not always cell 0) and one scenario is multi-batch
    # with a demand==capacity boundary, so a move_to that hardcodes a result,
    # only inspects the first group, or has an off-by-one in `== 0` will fail.
    #
    # Demands per scenario are integers [depot=0, c1, c2, c3] with
    # vehicle_capacity=_CAP and group_size=3. Expected tensors are hardcoded and
    # were verified by executing the reference solution `move_to`, including the
    # demand-equals-capacity boundary in `two-batch+eq` (a node whose demand
    # exactly equals remaining capacity stays reachable, since the solution masks
    # on `remaining_capacity < demand`).
    NEG = float("-inf")
    _MOVE_SCENARIOS = [
        dict(
            name="depot-first",
            demands=[[0, 3, 5, 7]],
            selected=[[0, 1, 3]],
            at_depot=[[True, False, False]],
            remaining=[[10, 7, 3]],
            mask=[[[NEG, 0, 0, 0], [0, NEG, 0, 0], [0, 0, NEG, NEG]]],
        ),
        dict(
            name="depot-mid+dup",
            demands=[[0, 3, 5, 7]],
            selected=[[1, 0, 0]],
            at_depot=[[False, True, True]],
            remaining=[[7, 10, 10]],
            mask=[[[0, NEG, 0, 0], [NEG, 0, 0, 0], [NEG, 0, 0, 0]]],
        ),
        dict(
            name="two-batch+eq",
            demands=[[0, 4, 6, 2], [0, 5, 1, 9]],
            selected=[[1, 2, 0], [0, 3, 2]],
            at_depot=[[False, False, True], [True, False, False]],
            remaining=[[6, 4, 10], [10, 1, 9]],
            mask=[[[0, NEG, 0, 0], [0, 0, NEG, 0], [NEG, 0, 0, 0]],
                  [[NEG, 0, 0, 0], [0, NEG, 0, NEG], [0, 0, NEG, 0]]],
        ),
    ]

    # Rebuild and re-run a fresh state *inside* each guarded `_safe` call, so an
    # exception during __init__ or move_to (e.g. a `...` blank or hardcoded
    # shape) becomes a hint instead of crashing the rest of the checks.
    def _run_move(scn):
        n_nodes = len(scn["demands"][0])
        data = _make_data(n_nodes, batch=len(scn["demands"]), device=DEVICE)
        int_demand = _make_int_demand(scn["demands"], device=DEVICE)
        st = GROUP_STATE(group_size=3, data=data, cost_matrix =_COST_MATRIX.to(DEVICE),
                         int_demand=int_demand, vehicle_capacity=_CAP)
        st.move_to(torch.LongTensor(scn["selected"]).to(DEVICE))
        return st

    # Default-arg binding (`s=scn`) captures each scenario by value, avoiding the
    # late-binding closure trap where every lambda would see the final scenario.
    for scn in _MOVE_SCENARIOS:
        nm = scn["name"]
        results.append(_safe(
            f"move_to[{nm}]: selected_count incremented to 1",
            "Increment self.selected_count by 1 (e.g. `self.selected_count += 1`).",
            lambda s=scn: _run_move(s).selected_count == 1,
        ))
        results.append(_safe(
            f"move_to[{nm}]: at_depot == (selected == 0)",
            "Set self.at_depot to a boolean tensor `selected_idx_mat == 0`.",
            lambda s=scn: torch.equal(
                _run_move(s).at_depot,
                torch.tensor(s["at_depot"], device=DEVICE),
            ),
        ))
        results.append(_safe(
            f"move_to[{nm}]: remaining_capacity decreased by demand, refilled at depot",
            "Subtract the selected demand from self.remaining_capacity, then set "
            "it to vehicle_capacity wherever at_depot.",
            lambda s=scn: torch.equal(
                _run_move(s).remaining_capacity,
                torch.tensor(s["remaining"], dtype=torch.long, device=DEVICE),
            ),
        ))
        results.append(_safe(
            f"move_to[{nm}]: ninf_mask masks visited + over-capacity nodes",
            "Set ninf_mask from the visited flags, plus -np.inf where demand "
            "exceeds remaining capacity (a node whose demand equals capacity "
            "stays reachable).",
            lambda s=scn: torch.equal(
                _run_move(s).ninf_mask,
                torch.tensor(s["mask"], device=DEVICE),
            ),
        ))

    return _summary("GROUP_STATE", results)


# --------------------------------------------------------------------------- #
# Part 4.2 — GROUP_ENVIRONMENT
# --------------------------------------------------------------------------- #
def _new_env(GROUP_ENVIRONMENT, DEVICE, finished=None):
    """Build a GROUP_ENVIRONMENT without calling __init__ (which needs feature
    tensors), wiring only the attributes the methods under test read."""
    env = GROUP_ENVIRONMENT.__new__(GROUP_ENVIRONMENT)
    env.cost_matrix = _COST_MATRIX.to(DEVICE)
    env.batch_s = 2
    env.group_s = 3
    env.group_state = _StubState(_SELECTED_NODES.to(DEVICE), finished=finished)
    return env


def test_group_environment(GROUP_ENVIRONMENT, DEVICE):
    """Check GROUP_ENVIRONMENT._get_travel_cost and GROUP_ENVIRONMENT.step."""
    print("Running tests for GROUP_ENVIRONMENT (Part 4.2)...")
    results = []
    expected_costs = _EXPECTED_COSTS.to(DEVICE)

    # ---- _get_travel_cost ----------------------------------------------- #
    cost_env = _new_env(GROUP_ENVIRONMENT, DEVICE)
    results.append(_safe(
        "_get_travel_cost: output shape is (batch, group)",
        "travel_costs should have shape (batch_s, group_s) after summing over the "
        "sequence dimension.",
        lambda: tuple(cost_env._get_travel_cost().shape) == (2, 3),
    ))
    results.append(_safe(
        "_get_travel_cost: total tour costs are correct",
        "Roll the node sequence (`from_nodes.roll(dims=2, shifts=-1)`), look up "
        "segment costs in the cost matrix, then sum over dim=2.",
        lambda: _allclose(cost_env._get_travel_cost(), expected_costs),
    ))

    # ---- step (all tours finished -> reward returned) ------------------- #
    finished_all = torch.ones((2, 3), dtype=torch.bool, device=DEVICE)
    done_env = _new_env(GROUP_ENVIRONMENT, DEVICE, finished=finished_all)
    action = torch.zeros((2, 3), dtype=torch.long, device=DEVICE)
    out_done = done_env.step(action)

    results.append(_safe(
        "step: returns a 3-tuple (group_state, reward, done)",
        "step should `return self.group_state, reward, done`.",
        lambda: isinstance(out_done, tuple) and len(out_done) == 3,
    ))
    results.append(_safe(
        "step: calls group_state.move_to with the selected actions",
        "Advance the state with `self.group_state.move_to(selected_idx_mat)`.",
        lambda: done_env.group_state.move_to_called
        and torch.equal(done_env.group_state.move_to_arg, action),
    ))
    results.append(_safe(
        "step: done is True when every tour is finished",
        "Aggregate per-tour flags into one boolean: `done = self.group_state.finished.all()`.",
        lambda: bool(out_done[2]),
    ))
    results.append(_safe(
        "step: reward is negative travel cost when done",
        "When done, set `reward = -self._get_travel_cost()`.",
        lambda: out_done[1] is not None and _allclose(out_done[1], -expected_costs),
    ))

    # ---- step (not all finished -> no reward yet) ----------------------- #
    finished_partial = torch.tensor([[True, True, True],
                                     [True, False, True]], device=DEVICE)
    notdone_env = _new_env(GROUP_ENVIRONMENT, DEVICE, finished=finished_partial)
    out_notdone = notdone_env.step(action)
    results.append(_safe(
        "step: done is False and reward is None while a tour is unfinished",
        "done must be False unless ALL tours are finished, and reward stays None "
        "until then.",
        lambda: (not bool(out_notdone[2])) and out_notdone[1] is None,
    ))

    return _summary("GROUP_ENVIRONMENT", results)
