# Sign-flip averages, input 2, hidden [2, 3, 1], first layer fixed to the identity

Exact averages over all 2^P sign patterns of layers 2..4 (P = 14).

| magnitudes | activation | nonconst_linear | const_linear | linear | const_activation | distinct_const_maps | mismatches |
|---|---|---|---|---|---|---|---|
| examples/quad_ring.json | 387/32 (12.093750) | 319/64 (4.984375) | 3105/2048 (1.516113) | 13313/2048 (6.500488) | 455/64 (7.109375) | 12159/8192 (1.484253) | 0 |
| examples/quad_axis.json | 387/32 (12.093750) | 319/64 (4.984375) | 3105/2048 (1.516113) | 13313/2048 (6.500488) | 455/64 (7.109375) | 12153/8192 (1.483521) | 0 |
| examples/quad_random0.json | 387/32 (12.093750) | 319/64 (4.984375) | 3107/2048 (1.517090) | 13315/2048 (6.501465) | 455/64 (7.109375) | 12195/8192 (1.488647) | 0 |
| examples/quad_random1.json | 387/32 (12.093750) | 319/64 (4.984375) | 97/64 (1.515625) | 13/2 (6.500000) | 455/64 (7.109375) | 6123/4096 (1.494873) | 0 |
| examples/quad_random2.json | 387/32 (12.093750) | 319/64 (4.984375) | 97/64 (1.515625) | 13/2 (6.500000) | 455/64 (7.109375) | 769/512 (1.501953) | 0 |

Columns: activation = activation regions; nonconst_linear / const_linear = linear regions on which the
network is non-constant / constant; linear = their sum; const_activation = activation regions with constant map;
distinct_const_maps = distinct constant values realised.  mismatches = networks where the independent polygon counter disagrees.
