# Project 6: EKF

## To Build

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt output.txt`

## Result

Dataset 1: 

```python
Accuracy - RMSE:
0.0689231
0.0638139
 0.556916
 0.553834
```

Dataset 2:

```python
Accuracy - RMSE:
0.185322
0.190148
 0.47124
0.775925
```
