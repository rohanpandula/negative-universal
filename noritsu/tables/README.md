# Recovered CorrectServ data

These files contain derived numeric data from the Noritsu `CorrectServ` tree.
They are not film-stock profiles.

The source hierarchy selects quality group/type, negative/positive, format
(135/120/240/110), and color/B&W. `ImgQltMrg.xml` maps Nega to quality types 0
and 1. The four supported Nega 135 selections are therefore group 0/1 × type
0/1; code rejects the old unsupported type-5 default.

## Files

| File | Status |
| --- | --- |
| `vendor_nega_135.json` | executable, verified extraction used by `noritsu.tables` |
| `base_tone_curve.json` | legacy compatibility data; not sufficient to reproduce the vendor curve builder |
| `per_film_tables.json` | historical reference only; no production path loads it |

`vendor_nega_135.json` records:

- the exact R/G/B 4096-entry Nega FilmLUT;
- three CommonCalc X/Y 20-point control pairs;
- the normal-output `ExecPattern=5` selector vector;
- source sizes and SHA-256 hashes.

The first CommonCalc 20-word block is X coordinates, not output values. A real
final gamma curve also requires `pcb_MakeGammaCurve2`, negative-tone anchoring,
and per-job controller inputs; direct interpolation is only a diagnostic.

## Read an installed table tree

Set `NORITSU_TABLES` to the directory containing `IQGrp000`, `IQGrp001`, and
their variant subdirectories:

```sh
NORITSU_TABLES=/path/to/ImageDataProc/Data/CorrectServ \
  python -m noritsu.audit ...
```

When `NORITSU_TABLES` is set, the loader consumes the verified struct sizes and
records the actual selected paths, file sizes, and hashes in the audit
manifest. The checked-in decoded artifact records source filenames, sizes, and
hashes without claiming a live filesystem path. Returned arrays and provenance
records are independent copies, so callers cannot mutate cached state.
