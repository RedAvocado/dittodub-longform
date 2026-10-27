"""Instrumentation only: count entries into mx.compile's traced Python body.

The counter includes unsuccessful trace attempts. It excludes eager priming and
flow's compiled=False path; compilation must remain enabled in MLX. It does not
count Metal shader libraries, dispatches, or successful device captures.
"""


def counted_compile(execute, stats):
    import mlx.core as mx

    def traced(*args):
        stats['compile_traces'] += 1
        return execute(*args)

    return mx.compile(traced)
