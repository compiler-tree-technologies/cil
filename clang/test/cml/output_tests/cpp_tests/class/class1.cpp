// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
public:
  int a;
  int b;
};
int main() {
  A obj1;
  obj1.a = 10;
  printf("%d\n", obj1.a); // CHECK: 10
  return 0;
}
