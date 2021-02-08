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
  B o2;
};

int main() {
  A o;
  o.o2.set_b(10);
  printf("%d\n", o.o2.get_b()); // CHECK: 10
  return 0;
}
