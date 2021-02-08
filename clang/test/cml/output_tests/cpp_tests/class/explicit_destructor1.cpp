// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>
class A {
  int val;
  float val2;

public:
  A(int a, float b) : val(a), val2(b) {}
  ~A() {
   printf("\nCalling the destructor");
  }
  int get_val() { return val; }
  float get_val2() { return val2; }
};

int main() {
  A o1(10, 13.2);
  int a = o1.get_val();
  a = a + 1;
  printf("%d\n", a); // CHECK: 11
  float b = o1.get_val2();
  printf("%f\n", b); // CHECK: 13.2
  o1.~A(); // CHECK: destructor
  return 0;
}
