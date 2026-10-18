# Numba acceleration benchmark summary

## Full optimizations (warm JIT, median [IQR] seconds; Python peak MB)

| case | numpy-old | numpy-new | numba-old | numba-new |
|---|---|---|---|---|
| nsgaii-zdt1-m2-p100 | 0.528 [0.528-0.531]; 0.7 | 0.469 [0.468-0.493]; 0.7 | 0.134 [0.133-0.135]; 0.4 | 0.078 [0.078-0.079]; 0.3 |
| nsgaii-zdt1-m2-p1000 | 6.115 [6.094-6.218]; 10.3 | 5.551 [5.533-5.551]; 10.3 | 0.076 [0.071-0.080]; 3.4 | 0.037 [0.037-0.039]; 2.3 |
| nsgaii-dtlz2-m3-p100 | 0.556 [0.548-0.563]; 0.6 | 0.477 [0.465-0.477]; 0.6 | 0.182 [0.179-0.195]; 0.2 | 0.083 [0.083-0.086]; 0.1 |
| nsgaii-dtlz2-m3-p500 | 8.649 [8.567-8.723]; 6.9 | 8.088 [8.039-8.223]; 6.9 | 1.070 [1.058-1.072]; 0.8 | 0.237 [0.235-0.245]; 0.6 |
| nsgaii-dtlz2-m5-p200 | 1.140 [1.138-1.151]; 2.3 | 0.974 [0.974-0.983]; 2.3 | 0.298 [0.292-0.320]; 0.4 | 0.107 [0.107-0.108]; 0.3 |
| nsgaii-dtlz2-m10-p200 | 1.353 [1.335-1.383]; 4.1 | 1.145 [1.140-1.152]; 4.1 | 0.405 [0.404-0.407]; 0.5 | 0.195 [0.194-0.198]; 0.4 |
| nsgaii-c1dtlz1-m3-p100 | 0.376 [0.354-0.395]; 0.6 | 0.230 [0.227-0.290]; 0.6 | 0.124 [0.121-0.130]; 0.2 | 0.069 [0.069-0.074]; 0.1 |
| nsgaii-expensive-m3-p100 | 3.281 [3.232-3.296]; 1.2 | 3.019 [3.009-3.019]; 1.2 | 3.052 [2.969-3.081]; 1.1 | 2.930 [2.910-2.933]; 1.1 |
| nsgaii-ss-zdt1-m2-p100 | 3.126 [3.097-3.136]; 0.7 | 2.052 [2.051-2.058]; 0.7 | 1.412 [1.397-1.441]; 0.2 | 1.150 [1.140-1.151]; 0.1 |
| nsgaii-ss-dtlz2-m3-p100 | 2.891 [2.858-2.981]; 0.6 | 1.827 [1.808-1.830]; 0.6 | 1.347 [1.342-1.360]; 0.1 | 1.249 [1.244-1.254]; 0.1 |
| moead-zdt1-m2-p100 | 6.873 [6.853-7.083]; 0.6 | 6.610 [6.401-6.683]; 0.6 | 2.552 [2.528-2.574]; 0.5 | 1.607 [1.598-1.617]; 0.5 |
| moead-dtlz2-m3-p91 | 7.700 [7.679-7.764]; 0.6 | 7.682 [7.441-7.852]; 0.6 | 2.802 [2.764-2.853]; 0.6 | 1.858 [1.845-1.860]; 0.6 |
| moead-c1dtlz1-m3-p91 | 3.572 [3.540-3.574]; 0.6 | 3.591 [3.557-3.613]; 0.6 | 1.583 [1.543-1.615]; 0.6 | 1.066 [1.062-1.070]; 0.6 |
| nsgaiii-dtlz2-m3-p91 | 0.590 [0.580-0.594]; 0.6 | 0.595 [0.586-0.602]; 0.6 | 0.300 [0.299-0.310]; 0.4 | 0.228 [0.219-0.228]; 0.4 |
| smsemoa-zdt1-m2-p100 | 2.596 [2.574-2.601]; 0.7 | 1.503 [1.490-1.519]; 0.7 | 1.230 [1.221-1.247]; 0.1 | 1.130 [1.125-1.133]; 0.1 |
| smsemoa-dtlz2-m3-p100 | 2.369 [2.345-2.373]; 0.6 | 1.313 [1.310-1.316]; 0.6 | 1.012 [1.008-1.020]; 0.1 | 0.885 [0.876-0.889]; 0.1 |
| spea2-dtlz2-m3-p100 | 0.682 [0.678-0.686]; 2.0 | 0.692 [0.691-0.704]; 2.0 | 0.700 [0.692-0.702]; 2.0 | 0.386 [0.385-0.390]; 2.0 |
| ibea-zdt1-m2-p100 | 0.228 [0.226-0.230]; 1.2 | 0.227 [0.227-0.229]; 1.2 | 0.225 [0.225-0.230]; 1.2 | 0.142 [0.142-0.145]; 1.2 |
| smpso-dtlz2-m3-p100 | 0.241 [0.240-0.242]; 0.6 | 0.238 [0.232-0.239]; 0.6 | 0.246 [0.238-0.249]; 0.2 | 0.169 [0.166-0.172]; 0.2 |
| agemoea-dtlz2-m3-p100 | 0.607 [0.606-0.624]; 1.8 | 0.609 [0.597-0.641]; 1.8 | 0.445 [0.442-0.449]; 1.9 | 0.201 [0.201-0.203]; 1.8 |
| rvea-dtlz2-m3-p91 | 0.054 [0.052-0.054]; 0.6 | 0.056 [0.055-0.057]; 0.6 | 0.052 [0.052-0.054]; 0.4 | 0.038 [0.038-0.038]; 0.4 |

## Final-front agreement

Configurations whose final objective matrices are bitwise identical for every seed (digest of `result.F`).

| case | identical groups |
|---|---|
| nsgaii-zdt1-m2-p100 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-zdt1-m2-p1000 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-dtlz2-m3-p100 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-dtlz2-m3-p500 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-dtlz2-m5-p200 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-dtlz2-m10-p200 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-c1dtlz1-m3-p100 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-expensive-m3-p100 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-ss-zdt1-m2-p100 | numpy-old = numpy-new = numba-new; numba-old |
| nsgaii-ss-dtlz2-m3-p100 | numpy-old = numpy-new = numba-new; numba-old |
| moead-zdt1-m2-p100 | numpy-old = numpy-new = numba-old = numba-new |
| moead-dtlz2-m3-p91 | numpy-old = numpy-new = numba-old = numba-new |
| moead-c1dtlz1-m3-p91 | numpy-old = numpy-new = numba-old = numba-new |
| nsgaiii-dtlz2-m3-p91 | numpy-old = numpy-new = numba-new; numba-old |
| smsemoa-zdt1-m2-p100 | numpy-old = numpy-new = numba-new; numba-old |
| smsemoa-dtlz2-m3-p100 | numpy-old = numpy-new = numba-new; numba-old |
| spea2-dtlz2-m3-p100 | numpy-old = numpy-new = numba-old = numba-new |
| ibea-zdt1-m2-p100 | numpy-old = numpy-new = numba-new; numba-old |
| smpso-dtlz2-m3-p100 | numpy-old = numpy-new = numba-old = numba-new |
| agemoea-dtlz2-m3-p100 | numpy-old = numpy-new = numba-new; numba-old |
| rvea-dtlz2-m3-p91 | numpy-old = numpy-new = numba-old = numba-new |

## Startup (fresh process; median seconds)

| label | case | phase | process | import | first run | second run |
|---|---|---|---|---|---|---|
| numpy-old | nsgaii-zdt1 | cold | 1.11 | 0.84 | 0.08 | 0.06 |
| numpy-old | nsgaii-zdt1 | cached | 1.13 | 0.85 | 0.10 | 0.06 |
| numpy-old | nsgaii-dtlz2 | cold | 1.12 | 0.82 | 0.07 | 0.06 |
| numpy-old | nsgaii-dtlz2 | cached | 1.09 | 0.81 | 0.07 | 0.06 |
| numpy-old | moead-zdt1 | cold | 2.18 | 0.84 | 0.59 | 0.61 |
| numpy-old | moead-zdt1 | cached | 2.10 | 0.80 | 0.59 | 0.60 |
| numpy-new | nsgaii-zdt1 | cold | 1.11 | 0.83 | 0.07 | 0.05 |
| numpy-new | nsgaii-zdt1 | cached | 1.06 | 0.80 | 0.07 | 0.06 |
| numpy-new | nsgaii-dtlz2 | cold | 0.99 | 0.75 | 0.06 | 0.05 |
| numpy-new | nsgaii-dtlz2 | cached | 1.03 | 0.78 | 0.06 | 0.05 |
| numpy-new | moead-zdt1 | cold | 1.98 | 0.78 | 0.53 | 0.54 |
| numpy-new | moead-zdt1 | cached | 2.00 | 0.78 | 0.53 | 0.55 |
| numba-old | nsgaii-zdt1 | cold | 3.64 | 0.96 | 2.22 | 0.01 |
| numba-old | nsgaii-zdt1 | cached | 1.60 | 0.86 | 0.42 | 0.02 |
| numba-old | nsgaii-dtlz2 | cold | 3.56 | 0.84 | 2.30 | 0.05 |
| numba-old | nsgaii-dtlz2 | cached | 1.58 | 0.83 | 0.39 | 0.02 |
| numba-old | moead-zdt1 | cold | 4.18 | 0.83 | 2.72 | 0.24 |
| numba-old | moead-zdt1 | cached | 2.03 | 0.82 | 0.66 | 0.25 |
| numba-new | nsgaii-zdt1 | cold | 4.88 | 0.78 | 3.67 | 0.01 |
| numba-new | nsgaii-zdt1 | cached | 1.54 | 0.75 | 0.44 | 0.01 |
| numba-new | nsgaii-dtlz2 | cold | 4.36 | 0.78 | 3.23 | 0.01 |
| numba-new | nsgaii-dtlz2 | cached | 1.52 | 0.76 | 0.45 | 0.01 |
| numba-new | moead-zdt1 | cold | 4.36 | 0.75 | 3.05 | 0.16 |
| numba-new | moead-zdt1 | cached | 1.74 | 0.78 | 0.56 | 0.16 |

## Kernels (median microseconds)

| operation | size | numpy-old | numpy-new | numba-old | numba-new |
|---|---|---|---|---|---|
| nsga2_ranking | M=2 N=50 | 666.3 | 661.3 | 27.1 | 25.4 |
| nsga2_survival | M=2 N=50 | 660.5 | 663.0 | 84.8 | 31.6 |
| nsga2_ranking | M=2 N=100 | 1859.2 | 1861.4 | 36.7 | 38.0 |
| nsga2_survival | M=2 N=100 | 1920.9 | 2003.4 | 122.1 | 39.9 |
| nsga2_ranking | M=2 N=200 | 5876.5 | 5949.2 | 70.7 | 71.0 |
| nsga2_survival | M=2 N=200 | 6064.4 | 6015.8 | 197.5 | 59.8 |
| nsga2_ranking | M=2 N=500 | 34023.6 | 34839.4 | 213.1 | 212.0 |
| nsga2_survival | M=2 N=500 | 35102.9 | 34699.2 | 478.8 | 158.3 |
| nsga2_ranking | M=2 N=1000 | 259607.0 | 266911.3 | 429.9 | 446.6 |
| nsga2_survival | M=2 N=1000 | 262940.1 | 274887.3 | 948.9 | 315.2 |
| nsga2_ranking | M=3 N=50 | 559.3 | 570.1 | 76.9 | 18.5 |
| nsga2_survival | M=3 N=50 | 579.5 | 584.6 | 118.5 | 25.4 |
| nsga2_ranking | M=3 N=100 | 1650.0 | 1665.3 | 351.3 | 42.3 |
| nsga2_survival | M=3 N=100 | 1679.6 | 1690.2 | 406.1 | 39.2 |
| nsga2_ranking | M=3 N=200 | 5875.1 | 5816.9 | 1422.5 | 143.4 |
| nsga2_survival | M=3 N=200 | 5925.3 | 6045.1 | 1623.6 | 114.5 |
| nsga2_ranking | M=3 N=500 | 71353.6 | 69817.8 | 10840.4 | 493.4 |
| nsga2_survival | M=3 N=500 | 80252.3 | 70867.0 | 11671.0 | 382.4 |
| nsga2_ranking | M=3 N=1000 | 282021.3 | 281655.9 | 36623.2 | 1312.4 |
| nsga2_survival | M=3 N=1000 | 282275.4 | 283594.0 | 37163.1 | 1061.1 |
| nsga2_ranking | M=5 N=50 | 575.5 | 570.9 | 95.9 | 33.8 |
| nsga2_survival | M=5 N=50 | 590.5 | 595.3 | 141.5 | 33.8 |
| nsga2_ranking | M=5 N=100 | 1861.0 | 1851.6 | 468.7 | 117.5 |
| nsga2_survival | M=5 N=100 | 1879.3 | 1882.0 | 547.3 | 102.5 |
| nsga2_ranking | M=5 N=200 | 7023.0 | 6935.5 | 2049.7 | 353.7 |
| nsga2_survival | M=5 N=200 | 7853.9 | 7567.7 | 2066.2 | 291.7 |
| nsga2_ranking | M=5 N=500 | 87576.8 | 87422.5 | 10272.1 | 1425.6 |
| nsga2_survival | M=5 N=500 | 90041.1 | 88910.2 | 9760.1 | 1222.0 |
| nsga2_ranking | M=5 N=1000 | 350206.1 | 357400.4 | 38292.4 | 3953.8 |
| nsga2_survival | M=5 N=1000 | 351813.2 | 347285.4 | 41473.1 | 3589.5 |
| nsga2_ranking | M=10 N=50 | 595.2 | 605.7 | 117.6 | 96.9 |
| nsga2_survival | M=10 N=50 | 612.0 | 619.4 | 146.0 | 100.0 |
| nsga2_ranking | M=10 N=100 | 1954.8 | 1971.4 | 478.7 | 287.8 |
| nsga2_survival | M=10 N=100 | 2139.4 | 1972.8 | 546.7 | 297.2 |
| nsga2_ranking | M=10 N=200 | 7862.8 | 8086.4 | 1815.2 | 884.9 |
| nsga2_survival | M=10 N=200 | 8010.5 | 7851.8 | 1859.7 | 884.0 |
| nsga2_ranking | M=10 N=500 | 95875.0 | 94075.3 | 10275.4 | 4229.0 |
| nsga2_survival | M=10 N=500 | 95614.7 | 101582.3 | 11353.6 | 5119.1 |
| nsga2_ranking | M=10 N=1000 | 382103.4 | 377460.0 | 42661.2 | 14535.3 |
| nsga2_survival | M=10 N=1000 | 385504.6 | 409868.4 | 41816.5 | 14831.4 |
| sbx_crossover | n_var=12 N=1 | 208.6 | 50.8 | 6.3 | 8.8 |
| polynomial_mutation | n_var=12 N=1 | 29.1 | 28.9 | 29.1 | 4.0 |
| tournament_selection | n_var=12 N=1 | 29.4 | 30.3 | 14.3 | 15.9 |
| sbx_crossover | n_var=30 N=1 | 205.9 | 53.8 | 7.1 | 11.1 |
| polynomial_mutation | n_var=30 N=1 | 29.5 | 28.8 | 29.4 | 4.3 |
| sbx_crossover | n_var=12 N=50 | 119.7 | 119.3 | 124.8 | 41.2 |
| polynomial_mutation | n_var=12 N=50 | 51.8 | 53.2 | 53.2 | 14.2 |
| tournament_selection | n_var=12 N=50 | 30.8 | 30.9 | 15.3 | 16.0 |
| sbx_crossover | n_var=30 N=50 | 222.2 | 211.6 | 216.4 | 88.3 |
| polynomial_mutation | n_var=30 N=50 | 61.4 | 61.9 | 63.6 | 22.2 |
| sbx_crossover | n_var=12 N=100 | 179.9 | 186.1 | 176.8 | 70.4 |
| polynomial_mutation | n_var=12 N=100 | 66.5 | 65.8 | 68.2 | 24.5 |
| tournament_selection | n_var=12 N=100 | 32.2 | 32.6 | 16.7 | 18.9 |
| sbx_crossover | n_var=30 N=100 | 353.3 | 359.8 | 354.5 | 178.1 |
| polynomial_mutation | n_var=30 N=100 | 84.5 | 84.8 | 86.2 | 43.0 |
| sbx_crossover | n_var=12 N=200 | 300.5 | 298.5 | 298.5 | 146.3 |
| polynomial_mutation | n_var=12 N=200 | 94.3 | 94.7 | 95.8 | 45.8 |
| tournament_selection | n_var=12 N=200 | 35.2 | 35.1 | 19.5 | 21.9 |
| sbx_crossover | n_var=30 N=200 | 657.6 | 675.1 | 643.4 | 305.9 |
| polynomial_mutation | n_var=30 N=200 | 131.7 | 132.2 | 134.5 | 73.7 |
| sbx_crossover | n_var=12 N=500 | 654.6 | 659.3 | 654.7 | 325.3 |
| polynomial_mutation | n_var=12 N=500 | 178.2 | 180.6 | 177.7 | 100.8 |
| tournament_selection | n_var=12 N=500 | 43.2 | 43.1 | 27.6 | 31.4 |
| sbx_crossover | n_var=30 N=500 | 1614.0 | 1648.1 | 1532.5 | 736.7 |
| polynomial_mutation | n_var=30 N=500 | 279.1 | 295.7 | 280.1 | 161.2 |
| sbx_crossover | n_var=12 N=1000 | 1269.1 | 1387.2 | 1283.9 | 543.0 |
| polynomial_mutation | n_var=12 N=1000 | 314.4 | 318.0 | 311.7 | 173.7 |
| tournament_selection | n_var=12 N=1000 | 55.6 | 54.9 | 41.1 | 37.6 |
| sbx_crossover | n_var=30 N=1000 | 4715.4 | 4787.5 | 4097.7 | 1346.6 |
| polynomial_mutation | n_var=30 N=1000 | 518.0 | 525.5 | 559.1 | 321.0 |
