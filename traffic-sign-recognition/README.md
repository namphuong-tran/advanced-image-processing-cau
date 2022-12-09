Run the centralized LeNet model:
```python=
python main_nn.py --dataset traffic --iid --num_channels 3 --model LeNet --epochs 500 --gpu 0
```
Run the federated LeNet model:
 ```python=
python fed.py --dataset traffic --iid --num_channels 3 --model LeNet --epochs 10 --local_ep 2 --num_users 10 --gpu 0
```