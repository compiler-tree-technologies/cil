// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include <stdio.h>

class A {
  public:
    int a;
};

int main() {
  A obj[10];
  obj[3].a = 10;
  printf("%d\n",obj[3].a); // CHECK: 10
  return 0;
}
