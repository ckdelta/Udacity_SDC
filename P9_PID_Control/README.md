# CarND-Controls-PID
Self-Driving Car Engineer Nanodegree Program

---

## Demo
https://www.youtube.com/watch?v=rfwNAFdKEkk

## Reflection

1) PID components, it works as shown on the deo video above.
```cpp
  // P = proportional gain, the steering when hit edge
  this->Kp = Kp;
  // I = Integral gain, steer to overcome the wrong direction
  this->Ki = Ki;
  // D = derivative gain, designed to reduce cte,
  this->Kd = Kd;
 ```
 
 2) Hyperparameters tuning
 
 The final parameters I chose are:
```
  double Kp = 0.1;
  double Ki = 0.001;
  double Kd = 2.0;
```
 I tried with twiddle first, but found out later by observing the car behavior on the track, it is easier to manually tune it. If the car steers too much at the eadge, then decrease I, otherwise increase I. If the car turns not smoothly, then decrease D, otherwise increase it. P also works to make turning smooth, but not that dramaically.
 
