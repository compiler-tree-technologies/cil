// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
  public:
    A() : a(1) {}
    int a;
};

void some_func(A o[10][10]) {
  o[1][1].a = 11;
}

int main() {
  A obj[10][10];
  obj[3][6].a = 10;
  printf("%d %d\n",obj[3][6].a, obj[9][1].a); // CHECK: 10 1
  some_func(obj);
  printf("%d %d\n",obj[3][6].a, obj[1][1].a); // CHECK: 10 11
  return 0;
}
