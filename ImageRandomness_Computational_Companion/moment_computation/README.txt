Exact moment computation
Algorithms, independent verification and computational comparisons

Model

The raw statistic is

    S = sum_{(i,j,c) in edges} c * abs(D_i - D_j).

For fixed references, D_i = abs(v_i-U_i), where v_i are deterministic and
U_i are mutually independent uniform values on range(q). Graph weights are
integers. Alphabet-enumerating interfaces also accept known independent
nonuniform probability laws. The output is the exact mean and variance or
the corresponding float64 evaluation, depending on the interface.

Inventory

COMPUTATIONAL_DETAILS.txt         Human-readable computational evidence and map
code/moment_engine.py             Exact breakpoint computation
code/alphabet_baseline.py         Exact and float64 alphabet baselines
code/reference_engine.py          Independent regression implementation
code/verify_algorithms.py         Exact and floating verification
code/run_comparison.py            Sequential timing and memory experiment
code/monte_carlo.py               Replicated simulation accuracy study
results/protocol.json             Prespecified experimental design and seeds
results/verification.json         Independent verification results
results/comparison.json           Full comparisons and timing repetitions
results/comparison.csv            Tabular computational results
results/monte_carlo.json           Full replicated simulation results
results/monte_carlo_replicates.csv Individual simulation runs
results/monte_carlo_summary.csv    Empirical error summaries
code/run_scaling.py               Extended degree/alphabet/area comparisons
code/vertex_ablation.py           Exact incident-projection kernel comparison
results/scaling_protocol.json     Prespecified extension and resource rules
results/scaling.json              Full extension records and exact outputs
results/scaling.csv               Per-case timing and accuracy summaries
results/vertex_scaling.csv        Central-vertex timing and structural counts
results/scaling_memory.csv        Separate peak traced-allocation records
analysis/verify_scaling_record.py Independent audit of saved scaling records
results/scaling_verification.json Saved-record arithmetic and provenance checks
analysis/run_vertex_benchmark.py  Six-degree vertex-only timing runner
analysis/verify_vertex_record.py  Read-only audit of the complete vertex batch
results/vertex_protocol.json      Prespecified vertex protocol
results/vertex_benchmark.json/csv  All 60 calls, exact outputs and complete medians
results/vertex_verification.json  Independent saved-record audit

COMPUTATIONAL_DETAILS.txt brings together the original and principal runtime
tables, complete area, window-width, alphabet and reference-diversity series,
graph-preparation costs, allocation records, replicated Monte Carlo errors
and execution provenance. Section 6 gives the complete principal benchmark
protocol, including case construction, processor and thread settings,
timing boundaries, fresh-call rotation, resource limits and the completion
ledger. Section 9 gives all principal comparison medians,
class-pair counts, event and interval counts, and unreduced output lengths.
It maps each result to the preserved files and keeps the timing protocols
separate.
The principal moment comparisons use the 18-case scaling record.
The complete six-degree batch supplies the vertex comparison.

The original 23 computational cases vary grid size, alphabet, window geometry
and reference structure. The three exact methods return rational moments;
Float64 alphabet approximates the same finite-sum formulas. All four
retain shared-endpoint covariance through incident-projection aggregation.
Prescribed work and workspace limits determine omissions. An omitted row
has no measured runtime and is not evidence of a timeout or speedup.

The separate 18-case extension varies window width from 2 to 16 on a
32 by 32 grid, alphabet size from 2^8 to 2^32 on a 16 by 16 grid, and image
area through 256 by 256. Two diverse-reference cases supplement the palette
inputs. The five-point area curve reuses the alphabet_q2p8_16 case for
16 by 16 and the degree_w2_32 case for 32 by 32 within this study, followed
by its area_64, area_128 and area_256 cases.
Some inputs also occur in the original 23-case study, whose measurements
remain separate. Do not pool their timing repetitions or form cross-study
time ratios.

The extension replaces the original alphabet work-proxy cutoffs with a
512 MiB estimated additional-workspace guard and a nominal 120-second limit
per call. SIGALRM delivery can be deferred during native calls; actual elapsed
times are retained. Three fresh moment repetitions use rotating method order.
A median is reported only when all three complete. Workspace omissions and
censored timeouts are separate statuses. The palette formula can change its
folded-law class count with q, so the saved class counts are part of the design.

The principal study completes 67 of 72 case-method combinations in 201
moment-timing calls. Five combinations exceed the workspace guard, and no
moment-timing call times out. All 15 completed Exact alphabet comparisons
agree with both breakpoint methods. The 16 completed Float64 alphabet
comparisons have maximum absolute relative variance error
1.0297705359333676e-13. The separate allocation-tracing timeout is documented
in Section 4 of COMPUTATIONAL_DETAILS.txt and does not replace a timing
observation. The complete case and method ledgers are in its Section 6.

The complete vertex comparison uses degrees 8, 16, 32, 64, 128 and 256 at q=65536.
Both kernels receive identical prepared profiles and compute one central
vertex's raw second-moment sum. Five fresh repetitions compare an event
sweep with explicit diagonal and cross-product integration. This isolates
incident-projection aggregation, not complete moment-algorithm runtime.
All 60 calls complete, with exact agreement at each degree. The complete
batch is stored in vertex_benchmark.json/csv. The historical vertex_scaling.csv is retained
for provenance and is never pooled with this separate batch. The center reference
is q//3, leaf j has reference j*(q-1)//(2*(d+1)) for j=1,...,d, and all
incident weights equal one. Leaf folded radii are mutually distinct.
Integer diagnostics report unreduced output-numerator bit lengths, not the
largest intermediate operand. Event and interval counts are structural work,
not CPU instructions.

Reproduction

Make a working copy before executing these commands from this directory.
The computational programs write their corresponding result files. The
--verify-only command performs untimed checks without starting measurements.
Python 3.12.14 and NumPy 2.3.5 were used for the recorded calculations.

    python3 code/verify_algorithms.py
    python3 code/run_scaling.py --verify-only
    python3 analysis/verify_scaling_record.py
    python3 code/run_comparison.py --authorized-protocol-sha256 cdde553f9cb220d2849314889de095b456a8d84016e740b571d41adff919cf54
    python3 code/monte_carlo.py --authorized-protocol-sha256 cdde553f9cb220d2849314889de095b456a8d84016e740b571d41adff919cf54

The extension is run with

    python3 code/run_scaling.py --authorized-protocol-sha256 8893cda7babdc2ad696ac36655e6175667c1556edde553cdb6580715c8c6eccf --phase all

This command resumes a matching existing record without repeating completed
or censored calls. To reproduce from fresh calls, use a fresh --output-dir
in the working copy. A source or protocol mismatch prevents silent resume.
The default phase order puts allocation tracing after all wall-clock timings.
The source and protocol hashes identify the implementations actually measured.

The complete vertex-only batch can be inspected or reproduced independently
of the 18 moment cases. The saved-record audit performs no kernel timings.

    python3 analysis/verify_vertex_record.py
    python3 analysis/run_vertex_benchmark.py --verify-only
    python3 analysis/run_vertex_benchmark.py --authorized-protocol-sha256 d7912ca64d3a0ba22acae67000b2c177d81a70877fe5799fcac88f2a20fc9d1c --output-dir results/vertex_reproduction

The fresh runner refuses to overwrite existing vertex result files. It
checkpoints every call and preserves every elapsed time, including unusually
slow calls. Ordinary automatic garbage collection remains enabled. Profile
preparation, explicit garbage collection and result serialization are outside
the timed kernels. Ratios use unrounded complete-batch medians.

Run all timing scripts sequentially, with other computational jobs
stopped where practical. They set backend thread variables before importing
NumPy. Each reported per-reference time includes fresh profile and pair
preparation and materialization of both outputs. Graph enumeration and
validation are measured separately. Caches are rebuilt on each call.

Selected memory peaks are separate Python traced-allocation measurements.
They exclude the preexisting reference and prepared graph and include output
materialization. They are not process resident-set-size measurements.

Recorded execution history

All 18 moment cases completed before a duplicate-key error in the vertex
CSV export interrupted the first degree-eight sweep call. That call had no
saved time or output and is recorded as censored_interruption. Its remaining
repetitions were suppressed. The degree-eight pairwise kernel and both
kernels at the other five degrees continued after the CSV-export correction.
All original moment observations were retained without another timing run.

provenance/run_scaling_moment_record.py retains the original measured runner.
provenance/scaling_interruption.json retains the checkpoint before the CSV-export correction.
The final report preserves its initial metadata and explicitly records the
source transition to the updated runner. The algorithm implementations,
input design and protocol bytes were unchanged. The updated runner changes
CSV record merging and adds an explicit, audited resume option for source
transitions. Ordinary resume still rejects an unexplained source mismatch.

API example

With code/ on Python's import path, compute raw and normalized moments on a
reusable rectangular-window graph as follows.

    from moment_engine import (
        prepare_image_graph, moments_prepared, image_normalizer
    )
    M, N, a, b, q = 8, 10, 2, 3, 256
    reference = [(17*i + 5) % q for i in range(M*N)]
    design = prepare_image_graph(M, N, a, b)
    result = moments_prepared(reference, q, design, cache=False)
    print(result.mean, result.variance)
    divisor = image_normalizer(M, N, a, b, q)
    score_mean, score_variance = result.normalized(divisor)

The raw moments are exact fractions. A stored GraphDesign.normalizer never
changes a raw result. Apply result.normalized(design.normalizer) explicitly
when that normalizer is desired. For fully contained a by b windows, the
image-score divisor is

    (M-a+1)*(N-b+1)*binomial(a*b,2)*(q-1).

Recompute this divisor whenever the geometry or q changes. GraphDesign
rejects self-loops and duplicate unordered edge records, including duplicate
zero-weight records. It supports signed integer weights and omits validated
zero-weight edges. The default exact core is deterministic. cache=True adds
hash-based profile and pair caching with expected/amortized lookup costs.
Both modes return the same exact fractions.

    from alphabet_baseline import enumerate_exact, enumerate_float
    exact = enumerate_exact(reference, q, design)
    approximate = enumerate_float(reference, q, design)

The exact alphabet baseline uses integer prefix sums independently of the
breakpoint engine. The float64 baseline reports any clipped negative edge
residual and its magnitude. The clipping tolerance is a numerical diagnostic
rule, not a uniform error bound.

General independent laws can be supplied as follows.

    from fractions import Fraction as Q
    from moment_engine import prepare_graph
    from alphabet_baseline import moments_independent_exact, moments_independent
    laws = [[Q(1,3), Q(2,3)], [Q(3,4), Q(1,4)]]
    graph = prepare_graph(2, [(0,1,2)])
    exact_general = moments_independent_exact(laws, graph)
    float_general = moments_independent(laws, graph)

The exact interface accepts integers and Fraction entries, requires exact
unit row sums and performs no renormalization. The float64 interface checks
normalization within its declared tolerance, normalizes accepted rows and
reports the maximum row-sum adjustment.

Interpretation

The algorithmic operation bound counts arithmetic operations. Integer bit
lengths retain their dependence on q and weight magnitudes. Relative-error
summaries use the exact rational output as the reference. Zero or sufficiently
small reference variances are assessed by the stated absolute-error rule.

The Monte Carlo study uses ten independent replications at each sample count.
Its relative RMSE and descriptive bootstrap intervals quantify the observed
cost--accuracy tradeoff and do not certify a population precision target.
These moment simulations are distinct from the rejection-rate and detection
experiments in the neighboring statistical_validation directory.

Public export identity

The companion-root EXPORT_PROVENANCE.json distinguishes original experimental
digests from distributed-file digests. Descriptive metadata and local paths
are normalized; numerical observations and computational parameters retain
their original values. Public checksums must not be interpreted as original
experimental file checksums.
