## Conclusion

**The claim in `CLAIM.md` is false.** `least_loaded` always constructs a feasible schedule under the stated model, but it does not always minimize makespan. It is an input-order **list-scheduling heuristic**, with a tight worst-case approximation ratio of **\(3/2\)** for two identical machines.

I executed the existing function and compared its results with an exhaustive optimal-schedule oracle. **`allocator.py` was not modified**; its Git diff is empty.

## 1. Feasibility: yes

In `allocator.py:4–7`, each job is appended to exactly one machine’s assignment list, and that machine’s load increases by the job’s duration. Therefore, after every iteration:

- Every processed job occurrence has been assigned exactly once.
- Each recorded load equals the sum of its assigned durations.

Running each machine’s assigned jobs consecutively from time zero produces a valid, nonpreemptive schedule. The maximum load returned on line 8 is its makespan.

The selection on line 5 chooses the currently less-loaded machine, breaking ties in favor of machine 0. It neither reorders jobs nor revisits previous decisions.

These conclusions assume the stated positive-integer input domain; the function does not validate it.

## 2. Optimality: concrete, observed counterexamples

These are actual return values from the unmodified function:

| Input | Observed return `(makespan, assignments)` | Optimal assignment | Optimal makespan |
|---|---|---|---:|
| `[1, 1, 2]` | `(3, [[1, 2], [1]])` | `[[1, 1], [2]]` | 2 |
| `[2, 1, 1]` | `(2, [[2], [1, 1]])` | `[[2], [1, 1]]` | 2 |
| `[3, 3, 2, 2, 2]` | `(7, [[3, 2, 2], [3, 2]])` | `[[3, 3], [2, 2, 2]]` | 6 |

### Why these optima are proven

For total work \(S\), every two-machine schedule has makespan at least \(\lceil S/2\rceil\).

- For `[1, 1, 2]`, \(S=4\). The displayed partition achieves the lower bound 2, proving optimality. The function instead returns 3—**50% above optimum**.
- For `[3, 3, 2, 2, 2]`, \(S=12\). The displayed partition achieves the lower bound 6, while the function returns 7.

The first two rows show that **input order affects the result**. The third shows that **sorting in descending order would not make the algorithm exact**, either.

The three-job counterexample is minimal in job count: with at most two positive-duration jobs, the function is optimal.

### Additional exhaustive verification

I checked **all 5,461 ordered lists of lengths 0 through 6 with durations in `{1, 2, 3, 4}`**, enumerating every two-machine assignment to determine each optimum.

Observed results:

- All returned schedules passed feasibility and makespan-consistency checks.
- **2,302** returned makespans were suboptimal.
- The largest observed ratio was **1.5**, attained by `[1, 1, 2]`.

This bounded test supports—but does not replace—the proofs.

## 3. Precisely when is the returned schedule optimal?

Let \(S=\sum_i p_i\), and let \(C\) be the returned makespan. Every assignment corresponds to a subset \(A\) of the **indexed job occurrences**, so

\[
\mathrm{OPT}
=\min_A\max\left(\sum_{i\in A}p_i,\;S-\sum_{i\in A}p_i\right).
\]

Consequently, an exact necessary-and-sufficient condition is:

> **The returned schedule is optimal if and only if no attainable subset sum \(s\) satisfies**
> \[
> S-C<s<C.
> \]

Indeed, a strictly better schedule exists exactly when both \(s<C\) and \(S-s<C\). For integer durations, the forbidden interval is \(S-C+1\le s\le C-1\).

### Useful sufficient conditions

For nonempty input, define

\[
L=\max\left(\max_i p_i,\;\left\lceil S/2\right\rceil\right).
\]

Every schedule has makespan at least \(L\). Thus **if the function returns \(C=L\), it is optimal**. Equal machine loads, or integer loads differing by one, satisfy this certificate.

This certificate is sufficient, **not necessary**: I observed

```text
least_loaded([2, 2, 2]) == (4, [[2, 2], [2]])
```

Its optimum is 4 because one machine must receive at least two jobs, although \(L=3\).

Input classes guaranteeing optimality include:

- **At most two jobs**, including the empty list, which returns `(0, [[], []])`.
- **All durations equal to \(d\)**: the algorithm balances job counts, attaining the necessary makespan \(d\lceil n/2\rceil\).
- **A dominant largest job comes first**, with \(p_{\max}\ge S-p_{\max}\): all remaining jobs go to the other machine, giving makespan \(p_{\max}\), which is a lower bound. The order condition matters; `[1, 1, 2]` already has such a dominant job, but places it last and fails.

## 4. General quality guarantee: tight \(3/2\), not exactness

For a nonempty input, choose a machine whose final load is \(C\). Let \(p\) be its last assigned job and \(t\) its load immediately before receiving that job.

Because it was selected as least loaded, the other machine then had load at least \(t\). Hence

\[
S\ge 2t+p,
\qquad
C=t+p\le \frac S2+\frac p2.
\]

Since \(\mathrm{OPT}\ge S/2\) and \(\mathrm{OPT}\ge p\),

\[
\boxed{C\le \frac32\,\mathrm{OPT}}.
\]

The observed `[1, 1, 2]` counterexample attains equality, so the bound cannot be improved for arbitrary input order.

## 5. Scaling: efficient computation, persistent optimality gap

### Computational scaling

For \(n\) jobs, under unit-cost arithmetic:

- **Time: \(\Theta(n)\).** `min` and `index` inspect only two loads, and list appends take amortized constant time.
- **Returned assignment storage: \(\Theta(n)\).**
- **Auxiliary storage excluding the output: \(O(1)\)** machine counters and bookkeeping.

For unbounded Python integers, arithmetic is not constant-cost. With total duration \(S>0\), load arithmetic gives a conservative **\(O(n\log(S+1))\)** bit-operation bound. The counters require \(O(\log(S+1))\) bits beyond the output references.

### Solution quality does not necessarily improve with size

For every integer \(k\ge1\), consider

```text
[1] * (2*k) + [2*k]
```

The unit jobs first produce loads \((k,k)\); the final job produces \((3k,k)\). The optimum is \(2k\), obtained by separating the large job from all unit jobs. Thus the ratio remains \(3/2\) for arbitrarily large inputs.

Observed executions included:

| Jobs \(n=2k+1\) | Returned makespan | Proven optimum |
|---:|---:|---:|
| 3 | 3 | 2 |
| 5 | 6 | 4 |
| 11 | 15 | 10 |
| 101 | 150 | 100 |

In contrast, exact optimization for arbitrary binary-encoded integer durations is weakly NP-hard via **PARTITION**. A subset-sum dynamic program offers \(O(nS)\) time and \(O(S)\) space, which is pseudopolynomial. This function performs no such search.

**Bottom line:** the implementation is a fast, feasible, order-sensitive \(3/2\)-approximation—not an always-optimal allocator.
