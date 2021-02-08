// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a;
  a = 3;
  int b = 10;

  if (a > 5) {
    a = 2;
    if (b < 10) {
      b = 20;
    } else {
      b = 30;
    }
  } else if (a < 5) {
    a = 11;
    if (b <= 10) {
      b = 20;
    } else {
      b = 30;
    }
  }

  printf("%d %d\n",a, b);
  return 0;
}
//CHECK: 11 20
