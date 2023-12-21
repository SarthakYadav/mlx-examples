## Comparing PyTorch (MPS) and MLX throughput

-> Device: Macbook M1 Pro (16 GB)

### Benchmarks
Following table contains training and inference throughput (images/sec) for PyTorch ([main_pytorch.py](main_pytorch.py)) and MLX ([main.py](main.py)), averaged over 5 epochs.

| Model     	| PyTorch<br>Train 	| PyTorch<br>Inference 	| MLX<br>Train 	| MLX<br>Inference 	|
|-----------	|------------------	|----------------------	|--------------	|------------------	|
| resnet20  	| 4452.36±11.43    	| 16591.67±222.13      	| 416.69±0.40  	| 2301.69±5.71     	|
| resnet44  	| 2115.64±5.40     	| 7732.72±26.00       	| 184.05±0.74  	| 1028.34±2.34     	|
| resnet110 	| 839.60±3.06      	| 3157.57±9.49         	| 69.34±0.18   	| 403.51±6.77      	|

There's a clear, distinct gap in throughput between PyTorch and MLX, for both training and inference.
Training on PyTorch MPS is ~10-11x faster than MLX, while inference on PyTorch MPS is ~6x faster than MLX.

### Inference/Training ratio for PyTorch and MLX
The following table highlights how fast inference is compared to training for both PyTorch and MLX, highlighting relative inference/training throughput.

| Model     	| PyTorch<br>Inference/Train 	| MLX<br>Inference/Train 	|
|-----------	|----------------------------	|------------------------	|
| resnet20  	| 3.73                      	| 5.52                  	|
| resnet44  	| 3.66                      	| 5.59                  	|
| resnet110 	| 3.76                      	| 5.82                  	|

MLX does inference faster relative to training than PyTorch.
