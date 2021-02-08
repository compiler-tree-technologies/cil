// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class B {
  int b;

public:
  void set_b(int val) { b = val; }
  int get_b() { return b; }
};

class A {
public:
  B *o2;
  A(B *o2) : o2(o2) {}
};

int main() {
  B o2;
  A o(&o2);
  o.o2->set_b(10);
  printf("%d\n", o2.get_b()); // CHECK: 10
  return 0;
}
