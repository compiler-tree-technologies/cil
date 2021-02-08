// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
public:
  int a;
  int b;
  void func(int c) { b = a + 3 + c; }
  int get_a() { return a; }
};
int main() {
  A obj1;
  obj1.a = 10;
  printf("%d\n", obj1.get_a()); // CHECK: 10
  obj1.func(100);
  printf("%d\n", obj1.b); // CHECK: 113
  return 0;
}
