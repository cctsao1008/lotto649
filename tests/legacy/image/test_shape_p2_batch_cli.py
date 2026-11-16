"""P2 outputs and safe file names."""
from pathlib import Path
import json
import numpy as np
import pytest

from lotto7x7.shape_p2_batch_cli import output_paths, report_html


def test_output_paths_all_unique():
    paths=output_paths(Path('results/v017_v018_p2_batch.json'))
    assert len(paths)==7 and len(set(paths.values()))==7
    assert paths['EXPLORER_HTML'].name.endswith('_explorer.html')


def test_output_paths_reject_other_extension():
    with pytest.raises(ValueError):output_paths(Path('p2.csv'))


def test_html_has_limits():
    r={'prequential_log_scores':{'edge':{'mean_log_gain_per_draw_bits':-.01,
        'cumulative_log_likelihood_ratio':-12,'iid_refit_upper_tail_p':.7,
        'iid_refit_upper_tail_bonferroni_across_three_levels':1}},
       'ticket_backtest':{'strategies':{'uniform':{'total_bets':12,
        'total_match_at_least_3':1,'total_match_at_least_4':0,
        'total_match_at_least_5':0,'total_match_6':0,'iid_conditional_p_ge_3plus':.4}}},
       'rare_shape_watchlist':{'class':'P6_O1_D5','observed_total':3,
          'observed_in_retrospective_evaluation':2}}
    h=report_html(r)
    assert 'Walk-forward' in h and 'P6_O1_D5' in h
    assert 'conditional' in h.lower() and '1/13,983,816' in h
