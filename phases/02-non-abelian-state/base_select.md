# Base-select — reliability via free-running-L16 base selection (`base_select.py`, 18.5M)

Train K=8 base seeds (short {4,8,16}); rank by FREE-RUNNING L16 e2e; post-train (1500 steps, lr 3e-4) the top & bottom, 2 restarts each; eval to L256. post_reliability established H1 (base-quality gates; between/within L128 var 0.087/0.0002). `clean` = L16 >= 0.95. Floor = 0.20, success = L128 > 0.5.

**Base L16 distribution:** 0.97, 0.86, 0.77, 0.62, 0.37, 0.30, 0.27, 0.26  → 1/8 clean (L16≥0.95), p_clean≈0.12; train K≈23 bases for a clean one at 95% confidence.

| selected | base L16 | post | L16 | L32 | L64 | L128 | L256 | success |
|---|---|---|---|---|---|---|---|---|
| bottom (s7) | 0.26 | s0 | 0.25 | 0.24 | 0.11 | 0.22 | 0.21 | no |
| bottom (s7) | 0.26 | s1 | 0.25 | 0.22 | 0.09 | 0.21 | 0.20 | no |
| top (s0) | 0.97 | s0 | 1.00 | 1.00 | 0.99 | 0.93 | 0.40 | YES |
| top (s0) | 0.97 | s1 | 1.00 | 1.00 | 0.98 | 0.89 | 0.38 | YES |
