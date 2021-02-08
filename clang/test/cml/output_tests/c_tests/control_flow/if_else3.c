// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a;
  a = 3;
  if (a > 5) {
    a = 2;
  } else if (a < 3) {
    a = 10;
  } else {
    a = 100;
  }

  printf("%d\n",a);
  return 0;
}
//CHECK: 100
