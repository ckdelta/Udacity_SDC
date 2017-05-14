#include "PID.h"

using namespace std;

/*
* TODO: Complete the PID class.
*/

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
  pre_error_ = 0;
  integral_ = 0;
  // P = the steer when hit edge, proportional gain
  this->Kp = Kp;
  // I = steer to overcome the wrong direction, Integral gain
  this->Ki = Ki;
  // D = to reduce cte, derivative gain
  this->Kd = Kd;
}

void PID::UpdateError(double cte) {

  // Proportional term
  p_error = - Kp * cte;
  // Integral term
  integral_ += cte;
  i_error = - Ki * integral_;
  // Derivative term
  d_error = - Kd * (cte - pre_error_);
  // Save error to previous error
  pre_error_ = cte;

}

double PID::TotalError() {
  // Calculate total output
  double steer = p_error + i_error + d_error;
  // cout << output << endl;
  // Limit to [-1,1]
  if (steer > 1) {
    steer = 1;
  }
  else if (steer < -1) {
    steer = -1;
  }
  return steer;
}
