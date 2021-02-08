// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
public:
  int a;
  int b;
  void func(int c) { b = a + 3 + c; }
};
int main() {
  A obj1;
  obj1.a = 10;
  printf("%d\n", obj1.a); // CHECK: 10
  obj1.func(10);
  printf("%d\n", obj1.b); // CHECK: 23
  return 0;
}
