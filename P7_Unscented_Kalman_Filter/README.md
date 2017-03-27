# Project 7: UKF

## To Build

1. Clone this repo.
2. Make a build directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./ExtendedKF path/to/input.txt path/to/output.txt`. You can find
   some sample inputs in 'data/'.
    - eg. `./ExtendedKF ../data/sample-laser-radar-measurement-data-1.txt ../result/output.txt`
5. NIS visualization is NIS_vis.ipynb at top directory.

## Result

Dataset 1: 

```python
Accuracy - RMSE:
0.0726209
0.074172
0.605913
0.573694

NIS < 7.815: 84.54619787408013%
```
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P7_Unscented_Kalman_Filter/result/data1" alt="data1"/>

Dataset 2:

```python
Accuracy - RMSE:
0.190295
 0.18505
0.271651
0.372804

NIS < 7.815: 98.99497487437185%
```
<img src="https://github.com/ckdelta/Udacity_SDC/blob/master/P7_Unscented_Kalman_Filter/result/data2" alt="data2"/>
