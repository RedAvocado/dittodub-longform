"""Reading-position guard logic, without the model."""
from dittodub_longform.core import Settings, _TextGuard


class _Arg:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v


def run(positions, s=Settings()):
    g = _TextGuard(100, 250, s)
    for step, p in enumerate(positions):
        g.pending = _Arg(p)
        g.update(step)
    return g


def test_frontier_follows_steady_reading():
    g = run([i // 3 for i in range(300)])
    assert 95 <= g.frontier <= 100
    assert g.key[0] == int(g.frontier) - Settings().margin  # text behind the frontier is masked


def test_frontier_rate_is_capped():
    g = run([0] * 10 + [100] * 10)
    assert g.frontier <= 0.5 * 10 + 1


def test_far_landing_holds_and_does_not_move_frontier():
    s = Settings(rewind=0)
    steady = [i // 3 for i in range(150)]
    g = run(steady + [149], s)  # 99 tokens past the frontier
    assert g.hold_until == 150 + s.hold
    assert g.frontier < 52
    assert g.key[1] == int(g.frontier) + s.ahead  # text beyond frontier + ahead hidden
    assert g.veto_stop(151) and not g.veto_stop(150 + s.hold)


def test_far_landing_rewinds():
    s = Settings()
    g = run([i // 3 for i in range(150)], s)
    g.pending = _Arg(149)
    assert g.update(150) == s.rewind
    r = 150 - s.rewind + 1
    assert len(g.trace) == r and len(g.log) == r  # the rewound frames are forgotten
    assert g.hold_until == r + s.hold and g.key[1] == int(g.frontier) + s.ahead


def test_hold_expires():
    steady = [i // 3 for i in range(150)]
    g = run(steady + [149] + [50 + i // 3 for i in range(40)])
    assert g.key[1] is None
