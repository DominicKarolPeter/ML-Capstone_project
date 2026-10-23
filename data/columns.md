# Column guide

Short meanings for the columns the two NASA datasets use, in plain words.
Units: ppm = parts per million (10,000 ppm is a 1 percent dip), Earth radii and
solar radii are relative to Earth and the Sun, K = kelvin, pc = parsec (about
3.26 light years).

## Classification: Kepler Objects of Interest (`data/kepler_koi.csv`)

Every row is one signal: a star where the Kepler telescope saw regular dips in
brightness. 9,564 rows, 153 columns; the notebook uses 99 of them as features.

**Target:** `koi_disposition`, NASA's verdict: CONFIRMED, CANDIDATE or FALSE POSITIVE.

**Dropped before modelling** (they contain the answer, or are not measurements):

| Column | Why it is dropped |
|---|---|
| `koi_score` | NASA's confidence score for the verdict |
| `koi_pdisposition` | the pipeline's own verdict |
| `koi_fpflag_nt`, `koi_fpflag_ss`, `koi_fpflag_co`, `koi_fpflag_ec` | false-positive flags: not transit-like, stellar eclipse, centroid offset, ephemeris match |
| `kepid`, `kepoi_name`, `kepler_name`, `koi_tce_delivname` | ids, names and text |

### The dip itself

| Column | Meaning |
|---|---|
| `koi_period` | orbital period: days between two dips, the planet's year |
| `koi_time0bk`, `koi_time0` | time of the first dip, in two time systems |
| `koi_duration` | how long one dip lasts, hours |
| `koi_depth` | how much the star dims during the dip, ppm |
| `koi_impact` | how central the crossing is: 0 across the middle of the star, 1 grazing the edge |
| `koi_model_snr` | signal-to-noise of the fitted dip: how clearly it stands out |
| `koi_num_transits` | number of dips observed |
| `koi_max_sngle_ev`, `koi_max_mult_ev` | strength of the strongest single dip, and of all dips combined |
| `koi_bin_oedp_sig` | odd-even test: whether alternate dips differ in depth, a sign of two stars eclipsing |
| `koi_ror` | planet radius divided by star radius |
| `koi_dor` | planet-to-star distance divided by star radius |
| `koi_ldm_coeff1`, `koi_ldm_coeff2` | limb-darkening coefficients used in the fit (the star's edge is dimmer than its centre) |
| `koi_count` | number of candidate planets found around that star |
| `koi_tce_plnt_num` | which planet number this signal is for its star |
| `transit_depth_per_hour` | our engineered feature: depth divided by duration |

### The planet, implied from the dip

| Column | Meaning |
|---|---|
| `koi_prad` | planet radius, Earth radii |
| `koi_sma` | size of the orbit (semi-major axis), astronomical units |
| `koi_incl` | tilt of the orbit, degrees |
| `koi_teq` | equilibrium temperature of the planet, K |
| `koi_insol` | insolation: starlight received, Earth = 1 |

### The star

| Column | Meaning |
|---|---|
| `koi_steff` | star temperature, K (the Sun is about 5,800) |
| `koi_slogg` | surface gravity, log scale (the Sun is 4.44) |
| `koi_smet` | metallicity, how much of the star is heavier than helium |
| `koi_srad` | star radius, solar radii |
| `koi_smass` | star mass, solar masses |
| `koi_srho` | star density from the transit fit |
| `koi_kepmag` | brightness in the Kepler band; a bigger number is fainter |
| `koi_gmag`, `koi_rmag`, `koi_imag`, `koi_zmag`, `koi_jmag`, `koi_hmag`, `koi_kmag` | brightness in other colour bands |
| `ra`, `dec` | position on the sky |

### Centroid columns: where the light comes from during the dip

A planet does not move the star's light centre; a background eclipsing star
does. These are the features the Random Forest used most.

| Column | Meaning |
|---|---|
| `koi_fwm_stat_sig` | how significant the shift of the light centre is |
| `koi_fwm_sra`, `koi_fwm_sdec` | position of the light centre during the dip |
| `koi_fwm_srao`, `koi_fwm_sdeco` | its shift from the out-of-dip position |
| `koi_fwm_prao`, `koi_fwm_pdeco` | shift of the dip's source from the target star |
| `koi_dicco_mra`, `koi_dicco_mdec`, `koi_dicco_msky` | offset of the dip's source from the out-of-dip centroid, in RA, Dec and on the sky |
| `koi_dikco_mra`, `koi_dikco_mdec`, `koi_dikco_msky` | the same offset measured from the star's catalogue position |

### Uncertainty columns

Anything ending in `_err1` or `_err2` is the plus or minus uncertainty of the
column it belongs to; `_err` on the centroid columns is their uncertainty.
Confirmed planets get more follow-up observations and are measured more
precisely, which is why some uncertainty columns (for example `koi_smet_err2`)
rank high in the feature importances.

### The twelve inputs on the app's classifier page

`koi_period`, `koi_duration`, `koi_depth`, `koi_prad`, `koi_model_snr`,
`koi_impact`, `koi_teq`, `koi_steff`, `koi_slogg`, `koi_srad`,
`koi_num_transits`, `koi_kepmag`. The other 87 features are filled with the
training medians.

## Clustering: TESS Objects of Interest (`data/tess_toi.csv`)

Every row is one signal found by the TESS telescope. 8,151 rows, 91 columns at
the 19 Oct 2026 download; the notebook uses 11 of them.

**Ids and the label, never used for fitting:**

| Column | Meaning |
|---|---|
| `toi`, `tid`, `toidisplay` | the object's id, its star's id, the display name |
| `tfopwg_disp` | the follow-up group's verdict, read only after clustering: PC planet candidate, CP confirmed planet, KP known planet, APC ambiguous candidate, FP false positive, FA false alarm |

**The eleven features** (log means the notebook takes log10 before clustering,
because the column spans several orders of magnitude):

| Column | Meaning |
|---|---|
| `pl_orbper` | orbital period, days (log) |
| `pl_trandurh` | transit duration, hours (log) |
| `pl_trandep` | transit depth, ppm (log) |
| `pl_rade` | planet radius, Earth radii (log) |
| `pl_insol` | insolation: starlight received, Earth = 1 (log) |
| `pl_eqt` | equilibrium temperature of the planet, K |
| `st_tmag` | TESS magnitude, star brightness; a bigger number is fainter |
| `st_dist` | distance to the star, pc (log) |
| `st_teff` | star temperature, K |
| `st_logg` | star surface gravity, log scale |
| `st_rad` | star radius, solar radii (log) |

**Not used:** sky position (`ra`, `dec`), proper motions (`st_pmra`, `st_pmdec`),
every `*err1`, `*err2`, `*lim` and `*symerr` column (uncertainties and flags),
the dates (`toi_created`, `rowupdate`, `release_date`), `pl_pnum` and the empty
`sectors` column.
