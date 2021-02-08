// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
public:
    int a;
};

int& func(A &obj1) {
  return obj1.a;
}
int main() {
  A obj1;
  obj1.a = 11;
  int &ref = func(obj1);
  ref++;
  printf("%d\n",obj1.a); // CHECK: 12
  return 0;
}
