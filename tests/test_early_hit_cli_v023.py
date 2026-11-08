"""CLI flags are explicitly wired, including a full 128-bit root seed."""
import argparse
from lotto7x7.early_hit_cli_v023 import configure_parsers


def test_cli_defaults_and_command_name():
    p=argparse.ArgumentParser();actions=p.add_subparsers(dest='command')
    configure_parsers(actions)
    a=p.parse_args(['early-hit-backtest'])
    assert a.trials==1000
    assert a.pool_size==10000
    assert a.cutoffs==[100,500,1000,5000]
    assert a.primary_cutoff==5000
    assert a.seed is None
    assert a.progress_every==0


def test_cli_preserves_exact_128bit_seed():
    p=argparse.ArgumentParser();actions=p.add_subparsers(dest='command')
    configure_parsers(actions)
    seed=210594425653336206127332942922757299723
    a=p.parse_args(['early-hit-backtest','--seed',str(seed),
                    '--trials','2000','--cutoffs','100','500','1000','5000',
                    '--progress-every','100'])
    assert a.seed==seed and a.trials==2000
    assert a.cutoffs==[100,500,1000,5000]
