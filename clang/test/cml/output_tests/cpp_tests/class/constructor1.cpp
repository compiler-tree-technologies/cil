// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>
class A {
  int val;

public:
  A(int a) : val(a) {}
  int get_val() { return val; }
};

int main() {
  A o1(10);
  int a = o1.get_val();
  a = a + 1;
  printf("%d\n", a); // CHECK: 11
  return 0;
}
