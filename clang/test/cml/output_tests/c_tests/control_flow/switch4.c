// RUN: %cml %s -o %t && %t | FileCheck %s

#include <stdio.h>
int main() {
  int a = 10, b = 1;
  int c = 19;
  switch (a) {
  default: {
    b = 10;
    while (c < 100) {
      if (c > 20)
        break;
      c++;
    }
    break;
  }
  case 1: {
    b = 3;
    if (b > 3) {
      a = a + 1;
    }
    break;
  }
  case 2: {
    b = 5;
    break;
  }
  case 5:
    b = 11;
    break;
  };
  printf("%d %d\n", b, c); // CHECK: 10 21
  return 0;
}
