// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>
class A {
public:
int func(int val) {
  return val;
}
};

int main() {
  A o1;
  int a = o1.func(10);
  a = a +1;
  printf("%d\n", a); // CHECK: 11
  return 0;
}
