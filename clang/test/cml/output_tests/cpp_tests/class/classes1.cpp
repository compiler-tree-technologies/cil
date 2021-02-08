// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class B {
public:
  int b;
};

class A {
public:
  int a;
};

int main() {
  A o;
  B o2;
  o.a = 10;
  o2.b = 11;
  printf("%d %d\n", o.a, o2.b); // CHECK: 10 11
  return 0;
}
