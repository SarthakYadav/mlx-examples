## Comparing PyTorch (MPS) and MLX throughput

-> Device: Macbook M1 Pro (16 GB)

### Flat out benchmarks
Following table contains training and inference throughput (images/sec) for PyTorch ([main_pytorch.py](main_pytorch.py)) and MLX ([main.py](main.py)), averaged over 5 epochs.

| Model     	| PyTorch<br>Train 	| PyTorch<br>Inference 	| MLX<br>Train 	| MLX<br>Inference 	|
|-----------	|------------------	|----------------------	|--------------	|------------------	|
| kwt1  	| 1485.61±7.42    	| 3928.82±40.31      	| 667.73±6.49  	| 3009.28±82.50     	|
| kwt2  	| 668.70±1.62     	| 1881.66±10.29       	| 395.56±5.10  	| 1495.28±38.46     	|

Training on PyTorch MPS is ~2x faster than MLX, while inference on PyTorch MPS is ~1.25x faster than MLX.

### Inference/Training ratio for PyTorch and MLX
The following table highlights how fast inference is compared to training for both PyTorch and MLX, highlighting relative inference/training throughput.

| Model     | PyTorch<br>Inference/Train | MLX<br>Inference/Train 	|
|-----------|----------------------------|------------------------	|
| kwt1  	| 2.64                       | 4.51                  	|
| kwt2  	| 2.81                       | 3.78                  	|

MLX does inference faster relative to training than PyTorch.
