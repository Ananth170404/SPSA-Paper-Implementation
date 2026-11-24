## Fig. 1 - didactic problems

| data set | variant | trials | epochs | E start (mean) | E final (mean) | E final (median) | train acc (mean) | trials >= 95 % acc | time |
|---|---|---|---|---|---|---|---|---|---|
| xor | bottom-up | 100 | 25 | 256.2 | 0.24 | 0.17 | 1.000 | 100 % | 10 s |
| xor | top-down | 100 | 25 | 256.2 | 0.22 | 0.18 | 1.000 | 100 % | 10 s |
| xor | random | 100 | 25 | 256.2 | 2.24 | 0.18 | 0.998 | 99 % | 27 s |
| xor | holistic | 100 | 25 | 256.2 | 28.27 | 0.04 | 0.965 | 86 % | 26 s |
| moons | bottom-up | 100 | 25 | 258.1 | 14.57 | 0.16 | 0.981 | 90 % | 11 s |
| moons | top-down | 100 | 25 | 258.1 | 16.40 | 0.13 | 0.979 | 88 % | 11 s |
| moons | random | 100 | 25 | 258.1 | 9.38 | 0.14 | 0.987 | 87 % | 24 s |
| moons | holistic | 100 | 25 | 258.1 | 111.77 | 74.87 | 0.858 | 31 % | 38 s |
| circles | bottom-up | 100 | 500 | 396.0 | 0.43 | 0.39 | 1.000 | 100 % | 471 s |
| circles | top-down | 100 | 500 | 396.0 | 0.45 | 0.42 | 1.000 | 100 % | 554 s |
| circles | random | 100 | 500 | 396.0 | 0.49 | 0.40 | 1.000 | 100 % | 1888 s |
| circles | holistic | 100 | 500 | 396.0 | 125.98 | 70.97 | 0.817 | 40 % | 1346 s |

## Fig. 2 and Sect. 4.2 - MNIST

| run | network (params) | method | order | split W/b | epochs | E start -> final (mean) | test acc at epoch 12 (per run) | final test acc (mean) |
|---|---|---|---|---|---|---|---|---|
| mnist | 784x784x196x10 (771,270) | SPSA | bottom-up | yes | 12 | 2.55 -> 2.78 | 0.481, 0.558, 0.505 | 0.515 |
| mnist | 784x784x196x10 (771,270) | GD | - | - | 250 | 2.46 -> 0.48 | 0.389 | 0.848 |
| mnist_155k | 784x196x10 (155,830) | SPSA | bottom-up | yes | 12 | 2.54 -> 2.01 | 0.519, 0.566, 0.558 | 0.548 |
| mnist_155k | 784x196x10 (155,830) | GD | - | - | 250 | 2.34 -> 0.55 | 0.352 | 0.805 |
| mnist_155k_holistic | 784x196x10 (155,830) | SPSA | holistic | yes | 12 | 2.54 -> 4.37 | 0.482, 0.493, 0.481 | 0.485 |
| mnist_155k_no_split_bias | 784x196x10 (155,830) | SPSA | bottom-up | no | 12 | 2.54 -> 1.98 | 0.559, 0.587, 0.530 | 0.558 |
| mnist_holistic | 784x784x196x10 (771,270) | SPSA | holistic | yes | 12 | 2.55 -> 5.60 | 0.465, 0.408, 0.527 | 0.467 |
| mnist_no_split_bias | 784x784x196x10 (771,270) | SPSA | bottom-up | no | 12 | 2.55 -> 2.71 | 0.524, 0.484, 0.572 | 0.527 |
