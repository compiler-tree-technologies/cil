// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
private:
  float d;

public:
  int a;
  int b;
  void func(int c) { b = a + 3 + c; }
  int get_a() { return a; }
  float get_d() { return d; }
  void set_c(float some) { d = some; }
};
int main() {
  A obj1;
  obj1.a = 10;
  printf("%d\n", obj1.get_a()); // CHECK: 10
  obj1.func(100);
  printf("%d\n", obj1.b); // CHECK: 113
  obj1.set_c(11.5);
  printf("%f\n", obj1.get_d()); // CHECK: 11.5
  return 0;
}
