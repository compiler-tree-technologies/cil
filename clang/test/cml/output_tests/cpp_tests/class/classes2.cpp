// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class B {
public:
  int b;
};

class A {
public:
  B o2;
};

int main() {
  A o;
  o.o2.b = 10;
  printf("%d\n", o.o2.b); // CHECK: 10
  return 0;
}
