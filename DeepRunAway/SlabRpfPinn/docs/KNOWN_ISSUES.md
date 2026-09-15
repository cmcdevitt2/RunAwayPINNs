# Known issues

Bugs and defects tracked for future fix. Not operating instructions — see
`CLAUDE.md`/`AGENTS.md` for those.

## `core/training.py` dead code (lines 1–~1595)

`core/training.py` locally defines `PinnDomain`, `PinnConfig`,
`make_pinn_config`, sobol/collocation helpers, `init_model`, `save_model`,
`load_model`, `probability`, `pde_coefficients`, and more in lines 1–~1595.
At line ~1596 the file re-imports the same names from `core.training_config`,
`core.model`, and `core.pde`, shadowing every local definition above that
point for the rest of the module. The first ~1595 lines are dead code.

Live definitions are in `core/training_config.py`, `core/model.py`, and
`core/pde.py`. Confirm with:

```bash
rg -n "^from core\." core/training.py
```

**Fix:** delete the shadowed local definitions in `core/training.py` (lines
1–~1595) and keep only the imports plus the training-loop code that uses
them.
