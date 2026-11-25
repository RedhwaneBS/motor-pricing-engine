# Data Provenance & Historical Scope: `beMTPL97`

## 1. Source & Integrity

- **Dataset**: `beMTPL97` (Belgian Motor Third-Party Liability insurance portfolio, 1997)
- **Origin**: `CASdatasets` R package (maintained by Christophe Dutang & Arthur Charpentier), originally analyzed in Denuit, Maréchal, Pitrebois & Walhin (*Actuarial Modelling of Claim Counts*, Wiley, 2007).
- **Upstream URL**: `https://github.com/dutangc/CASdatasets/raw/master/data/beMTPL97.rda`
- **License**: Distributed with `CASdatasets` under GPL (>= 2) for academic and educational research.
- **Immutability & Integrity**:
  - `data/raw/beMTPL97.rda` is treated as strictly **immutable** after acquisition.
  - Expected SHA-256 digest: `955a821a7a693bf18076c425e4d5a5a99889f3c89e1fbd99eca6239c11e963a6`.
  - Ingestion verifies the SHA-256 digest before parsing `.rda` into `data/processed/beMTPL97.csv`.

## 2. Portfolio Schema (163,212 policies, 18 variables)

| Column | Type | Role | Description |
| :--- | :--- | :--- | :--- |
| `id` | int | Identifier | Unique policy identifier (`1` to `163212`) |
| `expo` | float | Exposure | Policy duration in years (`0 < expo <= 1.0`, mean `0.890`) |
| `claim` | category | Target (excluded) | Binary claim indicator (`0` / `1`) |
| `nclaims` | int | Target (Frequency) | Observed claim count (`0` to `5`; 18,276 policies with `>= 1` claim) |
| `amount` | float | Target (Aggregate cost) | Total claim cost in EUR (`0.0` when `nclaims == 0`) |
| `average` | float | Target (Severity) | Mean cost per claim (`amount / nclaims` when `nclaims > 0`; `NaN` when `nclaims == 0`) |
| `coverage` | category | Feature | Coverage tier (`TPL`, `TPL+`, `TPL++`) |
| `ageph` | int | Feature (Spline) | Policyholder age in years (`18` to `95`) |
| `sex` | category | Feature | Policyholder gender (`male`, `female`) |
| `bm` | int | Feature | Statutory Belgian Bonus-Malus level in 1997 (`0` to `22`) |
| `power` | int | Feature | Vehicle horsepower (`10` to `243` kW) |
| `agec` | int | Feature | Vehicle age in years (`0` to `48`) |
| `fuel` | category | Feature | Fuel type (`gasoline`, `diesel`) |
| `use` | category | Feature | Vehicle usage (`private`, `work`) |
| `fleet` | category | Feature | Fleet flag (`0`, `1`) |
| `postcode` | int | Source feature | Belgian 4-digit postal code (`1000` to `9990`), mapped to `zone = postcode // 1000` (`1`..`9`) |
| `long`, `lat` | float | Spatial metadata | Commune centroid coordinates (not used directly in baseline GLM) |

## 3. Historical Context & Domain Limitations

**Important**: `beMTPL97` is a **1997 historical Belgian portfolio** and **must not be interpreted as representative of contemporary Belgian or European MTPL pricing**:

1. **Historical monetary & claim cost levels**: Monetary amounts reflect 1997 claim costs (converted to EUR) and historical bodily-injury/material-damage inflation regimes, not current repair, medical, or legal award levels.
2. **Abolished statutory Bonus-Malus (`bm`)**: The `bm` field (`0`–`22`) reflects Belgium's former statutory bonus-malus grid (which was abolished following EU liberalization in the early 2000s). Moreover, `bm` embeds prior claims experience (a posteriori rating), unlike pure a priori underwriting variables.
3. **Gender (`sex`) prohibition under EU law**: While `sex` is retained in this benchmark to faithfully illustrate multivariate confounding against one-way tables (e.g., *Werner & Modlin Ch. 9–10*), the **ECJ Test-Achats ruling** (effective December 21, 2012) prohibits gender-differentiated insurance pricing in the European Union.
4. **Portfolio composition & coverage changes**: Vehicle safety technology (ABS/AEB/ADAS), fleet electrification, and third-party liability limits have changed substantially since 1997.
