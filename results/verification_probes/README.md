# Truncation / regeneration probe records (2026-08-08 ~ 09)

Raw stdout logs are included where retained; probes whose logs were lost to a
session restart are transcribed below from the session record and are marked
as such. All probes can be re-run from the released code (the scripts fix
their own seeds/design indices, so every row below is reproducible).

## Included verbatim logs

- `aub_n21.log` — Au-B N=17 vs N=21 (is N=17 itself converged for gold?)
- `au_n13_probe.log` — Au-B N=13 class, N=13 vs N=17
- `cr_probe.log` — matched Cr control (A/B/C low-order cells, N=9 vs N=13)

## Transcribed (original stdout lost to a session restart)

Au-B N=9 class (N=9 vs N=13):
  idx10  501nm 0.0038 | 651nm 0.1772 | 780nm 0.0982
  idx28  501nm 0.0183 | 651nm 0.1697 | 780nm 0.0348
  idx29  501nm 0.0140 | 651nm 0.0295 | 780nm 0.0228
  idx31  501nm 0.0053 | 651nm 0.2512 | 780nm 0.2321

Au-A N=9@NIR class (archive N9 vs N13; <700nm cells are N13 in archive and
matched exactly):
  idx50  780nm 0.4034   idx51  780nm 0.2141   idx92  780nm 0.2320

Au-A N=17 control: idx0 @780nm, N=15 vs N=17 drift 0.0019 (archive = N17 rerun).

Regen flagged-sample discriminator (run twice in current env, run1-vs-run2 = 0
exactly for all four; deviations vs archive are deterministic cross-version):
  AuA idx56 0.0151 @764nm | AuA idx467 0.0186 @764nm
  AuB idx262 0.0057 @643nm | AuB idx462 0.1692 @542nm (isolated bad cell;
  neighbours reproduce to <=0.0001, see VERIFICATION.md §5)
