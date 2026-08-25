# Redesigned random controls: results

All conditions evaluated on the same 256 questions in the same order.
Intervals are 95% paired bootstrap over question indices, 10000 resamples.
Controls are disjoint from their binding sets by construction, so no
candidate pool is required and no z-score over control sets is reported.

## 1. Single-layer: binding against disjoint controls

Every value is a mean over the same 256 questions. `difference` is the
paired per-question contrast (binding minus control) with a 95% bootstrap
interval that resamples question indices once and applies them to both arms.


### Layer 10

condition                              drop   perturb difference         95% interval    d_z   %q>0
---------------------------------------------------------------------------------------------------
binding: causal ranks 1-200         +0.1515   0.01894  reference                                   
pass-through (0 features)           +0.0001   0.00119    +0.1513   [+0.1261, +0.1776]   0.72    82%
control: causal ranks 201-400       +0.0717   0.01239    +0.0798   [+0.0505, +0.1084]   0.34    67%
control: causal ranks 401-600       +0.0182   0.01131    +0.1333   [+0.1085, +0.1588]   0.65    80%
control: top-200 by activation      +0.1182   0.02360    +0.0332   [+0.0238, +0.0427]   0.43    73%
control: activation-matched         +0.0470   0.01820    +0.1045   [+0.0756, +0.1331]   0.45    70%

### Layer 11

condition                              drop   perturb difference         95% interval    d_z   %q>0
---------------------------------------------------------------------------------------------------
binding: causal ranks 1-200         +0.2131   0.01855  reference                                   
pass-through (0 features)           +0.0006   0.00136    +0.2125   [+0.1830, +0.2433]   0.85    86%
control: causal ranks 201-400       -0.0041   0.01844    +0.2172   [+0.1840, +0.2522]   0.78    83%
control: causal ranks 401-600       -0.0294   0.01356    +0.2425   [+0.2104, +0.2763]   0.88    84%
control: top-200 by activation      +0.1493   0.02652    +0.0637   [+0.0460, +0.0805]   0.46    71%
control: activation-matched         -0.0245   0.02418    +0.2376   [+0.2022, +0.2751]   0.79    83%

### Layer 12

condition                              drop   perturb difference         95% interval    d_z   %q>0
---------------------------------------------------------------------------------------------------
binding: causal ranks 1-200         +0.1672   0.02345  reference                                   
pass-through (0 features)           -0.0002   0.00121    +0.1674   [+0.1423, +0.1934]   0.79    84%
control: causal ranks 201-400       -0.0391   0.01639    +0.2063   [+0.1776, +0.2362]   0.86    86%
control: causal ranks 401-600       +0.0090   0.01490    +0.1582   [+0.1343, +0.1825]   0.80    84%
control: top-200 by activation      +0.1839   0.02956    -0.0167   [-0.0314, -0.0016]  -0.14    41%
control: activation-matched         +0.0137   0.02546    +0.1535   [+0.1287, +0.1785]   0.75    81%

### Layer 13

condition                              drop   perturb difference         95% interval    d_z   %q>0
---------------------------------------------------------------------------------------------------
binding: causal ranks 1-200         +0.1226   0.02023  reference                                   
pass-through (0 features)           -0.0000   0.00147    +0.1226   [+0.0883, +0.1607]   0.42    66%
control: causal ranks 201-400       +0.0069   0.02008    +0.1157   [+0.0770, +0.1576]   0.35    60%
control: causal ranks 401-600       +0.0272   0.01457    +0.0954   [+0.0596, +0.1347]   0.31    57%
control: top-200 by activation      +0.1048   0.02692    +0.0178   [+0.0072, +0.0289]   0.20    57%
control: activation-matched         -0.0136   0.02693    +0.1362   [+0.0974, +0.1793]   0.41    66%

### Layer 14

condition                              drop   perturb difference         95% interval    d_z   %q>0
---------------------------------------------------------------------------------------------------
binding: causal ranks 1-200         +0.2432   0.02450  reference                                   
pass-through (0 features)           -0.0004   0.00145    +0.2436   [+0.2174, +0.2701]   1.14    90%
control: causal ranks 201-400       -0.0132   0.01787    +0.2564   [+0.2307, +0.2820]   1.24    90%
control: causal ranks 401-600       +0.0147   0.01101    +0.2285   [+0.2007, +0.2567]   1.00    87%
control: top-200 by activation      +0.2038   0.03015    +0.0394   [+0.0291, +0.0498]   0.46    61%
control: activation-matched         -0.0071   0.02364    +0.2503   [+0.2232, +0.2771]   1.14    89%


## 2. Isolating the gradient term

causal_score = |gradient| x |activation|, so the causal and activation
rankings overlap heavily. These conditions ablate only the features on
which the two rankings disagree, plus the features they share.

condition                                n feat      drop   perturb
-------------------------------------------------------------------
L11: in both rankings                       141   +0.1642   0.01621
L11: causal-ranking only                     59   +0.0589   0.00770
L11: activation-ranking only                 59   -0.0277   0.01917
  L11 causal-only minus activation-only: +0.0866 [+0.0707, +0.1025]  (interval excludes zero)
L14: in both rankings                       154   +0.1672   0.02226
L14: causal-ranking only                     46   +0.0695   0.00819
L14: activation-ranking only                 46   +0.0297   0.01603
  L14 causal-only minus activation-only: +0.0399 [+0.0309, +0.0488]  (interval excludes zero)


## 3. Dose-response over causal rank (40 features per band)

Bands are disjoint and equal in size, so a decline cannot be a count effect.


### Layer 11

    causal ranks      drop   perturb  drop/perturb
--------------------------------------------------
            1-40   +0.1895   0.00912          20.8
           41-80   -0.0016   0.00748          -0.2
          81-120   +0.0065   0.00787           0.8
         121-160   -0.0003   0.00686          -0.0
         161-200   +0.0282   0.00762           3.7
         201-240   -0.0042   0.00651          -0.7
         241-280   +0.0103   0.00714           1.4
         281-320   +0.0121   0.00692           1.7
         321-360   -0.0264   0.00755          -3.5
         361-400   +0.0069   0.00738           0.9
  correlation of effect with band index: -0.543 (negative means the ranking orders features by effect)

### Layer 14

    causal ranks      drop   perturb  drop/perturb
--------------------------------------------------
            1-40   +0.1825   0.01227          14.9
           41-80   +0.1046   0.00939          11.1
          81-120   -0.0365   0.00923          -4.0
         121-160   -0.0178   0.00948          -1.9
         161-200   +0.0081   0.00980           0.8
         201-240   +0.0183   0.00900           2.0
         241-280   -0.0509   0.00819          -6.2
         281-320   -0.0133   0.00728          -1.8
         321-360   +0.0204   0.00767           2.7
         361-400   -0.0021   0.00639          -0.3
  correlation of effect with band index: -0.577 (negative means the ranking orders features by effect)


## 4. Random 40-feature subsets of two deep pools

Subsets of the top-200 and of ranks 201-1000 overlap by about 20% and 5%,
so between-set variance here is real, unlike the matched-control sets.

  random 40 of causal ranks 1-200        n=12  mean +0.0385  sd 0.0249  range [-0.0106, +0.0835]
  random 40 of causal ranks 201-1000     n=12  mean -0.0015  sd 0.0100  range [-0.0203, +0.0135]

  separation: top-200 subsets exceed tail subsets by +0.0400 on average; ranges overlap between the two sets of runs


## 5. Multi-layer conditions with symmetric controls

### 5a. The budget comparison (spreading against concentrating)

arm                               drop   control  adjusted   perturb
--------------------------------------------------------------------
spread 40 x 5 layers           +0.6182   +0.2122   +0.4059   0.01004
concentrated 200 at L11        +0.2131   -0.0041   +0.2172   0.01855

  raw ratio                : 2.90x
  control-adjusted ratio   : 1.87x [1.54, 2.26]

### 5b. Redundancy index with control-adjusted ablation

R = ablation / knockout. The knockout arm needs no control (it is an
attention mask, not a feature intervention), so only the numerator moves.

span           size        A  control    A adj        K   R raw   R adj        95% interval
-------------------------------------------------------------------------------------------
{14}              1  +0.2432  -0.0132  +0.2564  +0.3593   67.7%   71.4%    [ 65.6%,  77.6%]
{13,14}           2  +0.2679  +0.0007  +0.2672  +0.3191   83.9%   83.7%    [ 74.7%,  93.0%]
{12,13,14}        3  +0.5007  -0.0207  +0.5214  +0.5721   87.5%   91.1%    [ 82.8%, 100.3%]
{11-14}           4  +0.8302  -0.0150  +0.8452  +1.2762   65.1%   66.2%    [ 62.1%,  70.7%]
{10-14}           5  +1.0028  +0.0605  +0.9423  +1.3934   72.0%   67.6%    [ 63.9%,  71.5%]
{10,11,12}        3  +0.7590  +0.0596  +0.6994  +1.2975   58.5%   53.9%    [ 50.3%,  57.8%]
{10,12,14}        3  +0.5631  +0.0306  +0.5325  +0.7648   73.6%   69.6%    [ 65.1%,  74.2%]
  slope of R against span size, raw      : -0.0103 per layer
  slope of R against span size, adjusted : -0.0250 per layer
  (a redundancy account predicts a POSITIVE slope: wider spans should
   recover more of the knockout ceiling)

### 5c. Control magnitude across the remaining multi-layer conditions

condition                        binding   control  control share
-----------------------------------------------------------------
loo_drop10                       +0.8302   -0.0150          -1.8%
loo_drop11                       +0.6681   +0.0505           7.6%
loo_drop12                       +0.7037   +0.0644           9.2%
loo_drop13                       +0.9166   +0.0460           5.0%
loo_drop14                       +0.9066   +0.0681           7.5%
downstream_ablate_L11            +0.2131   -0.0041          -1.9%
downstream_ablate_L14            +0.2432   -0.0132          -5.4%
budget_concentrated_L11_k40      +0.1895   -0.0016          -0.9%
budget_concentrated_L11_k100     +0.1920   +0.0209          10.9%
budget_concentrated_L11_k200     +0.2131   -0.0041          -1.9%
budget_concentrated_L11_k400     +0.2364   -0.0431         -18.2%
budget_concentrated_L11_k800     +0.1940   -0.0045          -2.3%
