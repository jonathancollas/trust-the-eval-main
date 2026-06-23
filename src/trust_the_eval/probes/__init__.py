"""Probe registry. Importing this package self-registers all 20 probes.

Anatomy of an eval, attacked at each joint:
  items -> presentation -> behaviour -> scoring -> statistics -> time.
Every probe answers "can I trust this eval RESULT?" — never "how do I make a
model misbehave". See CONTRIBUTING.md for the scope gate.
"""
# EPIC 1 - items
from . import dataset_hygiene            # noqa: F401  [static]
from . import contamination_perturb      # noqa: F401  [model]
from . import label_error_audit          # noqa: F401  [model]
from . import item_ambiguity             # noqa: F401  [model]
from . import coverage_distribution      # noqa: F401  [static]
from . import discrimination_saturation  # noqa: F401  [static]
# EPIC 2 - presentation
from . import prompt_format_sensitivity  # noqa: F401  [model]
from . import option_order_bias          # noqa: F401  [model]
from . import answer_extraction_audit    # noqa: F401  [static]
# EPIC 3 - behaviour
from . import sandbagging_paired         # noqa: F401  [model]
from . import elicitation_ceiling        # noqa: F401  [model]
from . import refusal_confound           # noqa: F401  [model]
from . import self_consistency           # noqa: F401  [model]
from . import reward_hacking_eval        # noqa: F401  [static]
# EPIC 4 - scoring
from . import judge_swap                 # noqa: F401  [model]
# EPIC 5 - statistics
from . import statistical_power          # noqa: F401  [static]
from . import subgroup_power             # noqa: F401  [static]
from . import multiplicity_cherrypick    # noqa: F401  [static]
# EPIC 6 - time & provenance
from . import model_drift                # noqa: F401  [model]
from . import provenance_repro           # noqa: F401  [model]
