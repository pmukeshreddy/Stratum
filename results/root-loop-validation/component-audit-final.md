# Read-only specification audit

**All four components have demonstrated specification violations.** Checks ran under Python 3.12.12 with bytecode generation disabled. No files were changed: implementation hashes remained identical, `git status --short` and `git diff --exit-code` were clean, and no `__pycache__` directories were created.

## Reproduction setup

Run Python with `python3 -B` from the repository directory:

```python
from intervals import merge_intervals as merge, covered_length as length
from ledger import reconcile
from records import encode_record as enc, decode_record as dec
from ordering import stable_topological_order as topo

def E(account, ident, status, amount):
    return {
        "account": account,
        "id": ident,
        "status": status,
        "amount": amount,
    }
```

Outputs below use Python representations; `\n` denotes an actual newline in the value.

## 1. `intervals.py`

### Demonstrated violations

| Minimal check | Observed output | Required output |
|---|---|---|
| `merge([[0, 1], [1, 2]])` | `[[0, 1], [1, 2]]` | `[[0, 2]]`: merge adjacency |
| `merge([[2, 2]])` | `[[2, 2]]` | `[]`: ignore empty intervals |
| `merge([[0, 10], [2, 3]])` | `[[0, 3]]` | `[[0, 10]]`: nesting must not shrink coverage |
| `length([[0, 10], [2, 3]])` | `3` | `10`: return union length |

The empty-interval bug can also destroy existing coverage:

```python
merge([[0, 10], [2, 2]])   # observed: [[0, 2]]
length([[0, 10], [2, 2]])  # observed: 2; required: 10
```

**Cause:** `intervals.py:6` separates adjacent intervals using `>=`; no branch skips empty intervals; line 9 replaces the previous end instead of taking its maximum. `covered_length` inherits the corrupted merge result.

**Focused correction:** Reject reversed intervals, skip empty intervals, start a new interval only when `start > previous_end`, and otherwise extend with `max(previous_end, end)`. Continue allocating output lists rather than mutating input.

**Passing controls:** Reversed input `[[3, 2]]` raised `ValueError: reversed interval`; ordinary overlap `[[3, 6], [1, 4]]` produced `[[1, 6]]`; empty input produced `[]`. Input nonmutation held in the targeted and exhaustive checks below.

## 2. `ledger.py`

For this table, evaluate **`reconcile(events)`** with each listed event sequence.

### Demonstrated violations

| Events | Observed output | Required behavior |
|---|---|---|
| `[E('A','x','settled',4), E('B','x','settled',7)]` | `{'A': 4}` | `{'A': 4, 'B': 7}`: identity includes account |
| `[E('A','x','pending',4), E('A','x','settled',4)]` | `{}` | `{'A': 4}` |
| `[E('A','x','cancelled',4), E('A','x','settled',4)]` | `{}` | `{'A': 4}` |
| `[E('A','x','settled',4), E('A','x','settled',7)]` | `{'A': 4}` | Raise `ValueError` for conflicting settled amounts |
| `[E('A','x','settled','7')]` | `{'A': 7}` | Reject string amount |
| `[E('A','x','settled',7.9)]` | `{'A': 7}` | Reject float amount |
| `[E('A','x','settled',True)]` | `{'A': 1}` | Reject Boolean amount |

The specification does not prescribe an exception class for invalid amount types; the demonstrated defect is their acceptance and coercion.

**Causes:**
- `ledger.py:4` uses only `id`, not `(account, id)`.
- Lines 5–9 mark observations as seen before checking whether they settled.
- The `seen` set stores no amount, so conflicting settlements are silently skipped.
- Line 11 applies `int(...)`, accepting prohibited types.

**Focused correction:** Validate amounts without coercion—accept integers but explicitly exclude `bool`. Track applied settled amounts in a mapping keyed by `(account, id)`. Pending/cancelled observations must not populate that mapping. For an already-settled key, compare amounts: equal means no additional application; different means `ValueError`. Retain account entries when their balances reach zero.

**Passing controls:** Identical repeated settlements applied once. An applied zero amount returned `{'A': 0}`, as did separate `+4` and `-4` settlements. Pending/cancelled-only input returned `{}`.

## 3. `records.py`

### Demonstrated violations

| Minimal check | Observed output | Required behavior |
|---|---|---|
| `enc(['a,b'])` | `'a,b'` | Encode one field, e.g. `'"a,b"'` |
| `dec('"a,b"')` | `['a', 'b']` | `['a,b']` |
| `dec('"a""b"')` | `['a""b']` | `['a"b']` |
| `dec(enc(['"x"']))` | `['x']` | `['"x"']`: preserve literal quotes |
| `enc(['a\nb'])` | `'a\nb'` | Quote the multiline field, e.g. `'"a\nb"'` |
| `dec('"a')` | `['a']` | Reject unterminated quoting |
| `dec('"a"x')` | `['a"x']` | Reject junk after a closing quote |
| `dec('a\nb')` | `['a\nb']` | Reject multiple records |

CRLF-separated records are also accepted: `dec('a\r\nb')` returned `['a\r\nb']`.

These represent four underlying defects: **missing encoder escaping, incorrect quoted-field decoding, absent malformed-quoting validation, and absent record-count validation.**

**Independent confirmation:** Python’s standard-library CSV parser read the library’s `enc(['a\nb'])` output as **two records**, `[['a'], ['b']]`. It decoded `'"a""b"'` as `[['a"b']]`. With strict parsing, `'"a'` raised `csv.Error: unexpected end of data`, and `'"a"x'` raised `csv.Error: ',' expected after '"'`.

Importantly, the library’s own newline round trip passes:

```python
dec(enc(['a\nb']))  # observed: ['a\nb']
```

That does **not** establish compliance: the encoder emits multiple records and the decoder incorrectly treats them as one, masking the problem.

**Cause:** `records.py:2` uses `split(',')` and `strip('"')`, neither of which implements CSV parsing. Line 5 joins unescaped field strings.

**Focused correction:** Use standard-library `csv.writer` and newline-preserving `io.StringIO`. Remove exactly the writer’s final record terminator—not arbitrary trailing whitespace or newlines. Decode using strict CSV parsing, preserve whitespace, and enforce exactly one logical record. Add explicit quote-grammar validation for any malformed forms the chosen parser tolerates; `strict=True` alone is not a universal quote validator.

**Passing controls:** `['', '']` and `['  x  ', '雪', '']` round-tripped unchanged. Decoding `'"a\nb"'` worked. Returned fields were strings, and ordinary encoding added no terminal newline. There is no demonstrated *general* whitespace or Unicode defect.

## 4. `ordering.py`

### Demonstrated violations

| Minimal check | Observed output | Required behavior |
|---|---|---|
| `topo(['a','b'], [])` | `['b', 'a']` | `['a', 'b']`: earliest supplied eligible node first |
| `topo(['a','a'], [])` | `['a', 'a']` | Raise `ValueError` for duplicate nodes |
| `topo(['a'], [('x','a')])` | `KeyError: 'x'` | Raise `ValueError` for unknown parent |
| `topo(['a'], [('a','x')])` | `KeyError: 'x'` | Raise `ValueError` for unknown child |
| `topo(['a'], [('a','a')])` | `[]` | Raise `ValueError` for cycle |
| `topo(['a','b'], [('b','b')])` | `['a']` | Raise `ValueError`, not return a partial order |

Duplicate edges also change the result:

```python
nodes = ['a', 'b', 'c']
edges = [('a', 'c'), ('a', 'b')]

print(topo(nodes, edges))
print(topo(nodes, edges + [('a', 'c')]))
```

Observed:

```text
['a', 'b', 'c']
['a', 'c', 'b']
```

Both calls must return `['a', 'b', 'c']`. The baseline passes, but adding an already-present edge breaks the specified order.

**Causes:**
- `ordering.py:10` pops from a LIFO stack instead of selecting the earliest-ranked eligible node.
- Lines 2–6 omit duplicate-node and endpoint validation and count/store duplicate edges.
- Line 16 returns without checking whether traversal emitted every node.

The duplicate-edge failure interacts with LIFO scheduling. It is **not** evidence that duplicate edges necessarily produce duplicate output nodes or false cycle errors.

**Focused correction:** Validate nodes and endpoints, deduplicate edges, and use an original-node-index priority queue, such as `heapq` over indices. Raise `ValueError` if traversal does not emit every validated node.

Merely switching to FIFO is insufficient: for nodes `['a','b','c']` and edge `('b','a')`, the required order is `['b','a','c']`; an earlier-ranked node that becomes eligible must outrank nodes already waiting.

**Passing controls:** A simple chain and a duplicated single chain edge produced correct orders. Inputs remained unchanged in all targeted checks and the exhaustive graph check.

## Independent cross-checks and regression coverage

These checks ran in memory; no test files were added.

| Component | Independent check | Observed result |
|---|---|---|
| Intervals | Enumerated all 1,111 ordered sequences of 0–3 nonreversed intervals with endpoints −1 through 2. Oracle enumerated covered integer points and rebuilt contiguous runs. | **940 merge mismatches**, **208 length mismatches**, **0 input mutations** |
| Ledger | Enumerated 6,175 sequences of 0–3 events using two accounts, one ID, three statuses and amounts −1/0/1. Oracle grouped settled amounts by `(account,id)` before calculating balances. | **3,192 mismatches:** 600 missed conflicts and 2,592 incorrect balance results |
| Records | Tested 584 field sequences of lengths 1–3 drawn from empty, plain, whitespace, comma, quote, Unicode, LF and CR examples; compared decoding against standard CSV writer output. | **326 decoding mismatches**; the same corpus also had **326 library round-trip mismatches** |
| Ordering | Enumerated all 531 directed graphs on 0–3 nodes, including self-loops. Oracle used predecessor sets and scanned original node order at every step. | Of 30 DAGs, **14 wrong orders**; **all 501 cyclic graphs accepted**; **0 input mutations** |

These are bounded checks, not proofs over all possible inputs.

### Scope, ambiguities, and next steps

- **Standard-library-only requirement:** satisfied by source inspection; all four implementations currently use only built-ins.
- **Regression-test gap:** no test files are tracked, despite the specification’s test requirement. After corrections, preserve the minimal checks and bounded oracles above as regression tests.
- **CSV boundaries requiring clarification:** zero-field records versus a singleton empty field, and acceptance of an optional terminal record separator. For example, `enc([''])` returned `''`, which the standard CSV reader interpreted as no records, although the library reconstructed `['']`. I have not counted that policy boundary among the definitive defects above.
- I have not inferred additional requirements for unspecified input shapes, such as unhashable nodes or malformed event dictionaries.

The immediate correction priorities are **preventing silent data loss** in intervals and ledger, **making CSV output interoperable and input validation strict**, and **rejecting invalid graphs while enforcing the required topological priority**.
