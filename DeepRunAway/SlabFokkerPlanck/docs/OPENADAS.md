# OpenADAS data workflow

The future bulk-plasma model needs unresolved ADF11 data for:

- `SCD`: effective electron-impact ionization coefficients;
- `ACD`: effective recombination coefficients;
- `PLT`: line power from excitation;
- `PRB`: recombination and bremsstrahlung power.

OpenADAS files are user-owned inputs and are not redistributed by this
repository. The acquisition helper stores the requested files in a local,
untracked directory and writes a SHA-256 manifest containing the source URL,
retrieval time, file path, and OpenADAS terms URL:

```bash
source ../.venv/bin/activate
python scripts/fetch_openadas.py \
  --element Ar --year 89 \
  --data-dir data/openadas/Ar89
```

The available year and class combination must be checked on the
[OpenADAS ADF11 listing](https://open.adas.ac.uk/adf11). For example, the
current unresolved argon listing provides 1989 ACD/SCD/PLT/PRB files, while
not every element has every year or class.

`openadas_data.py` reads standard unresolved ADF11 files and interpolates
bilinearly in the native `log10(Te/eV)` and `log10(ne/cm^-3)` axes. Requests
outside the native rectangle fail; no extrapolation is permitted. ADF11
coefficients remain in their native units (`cm^3 s^-1` for ACD/SCD and
`W cm^3` for PLT/PRB), with the radiated-power helper converting the result to
`W m^-3`.

The current forward executable still uses its prescribed ion table. The
OpenADAS reader is the data foundation for the pending collisional-radiative,
bulk-energy, and induction coupling; it is not silently activated by the
existing TOML case.
