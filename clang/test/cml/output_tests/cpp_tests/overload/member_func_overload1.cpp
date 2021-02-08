// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>
class A {
  private:
    int some;
public:
int func(int val) {
  some = 3;
  return val;
}

float func(float val) {
  return val;
}
};
int main() {
  A o1;
  int a = o1.func(10);
  float val = 10.32;
  float b = o1.func(val);
  a = a +1;
  b = b  + 2.0;
  printf("%d\n", a); // CHECK: 11
  printf("%f\n", b); // CHECK: 12.32
  return 0;
}
