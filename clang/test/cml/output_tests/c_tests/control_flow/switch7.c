// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a[6] = {1, 2, 3, 4, 5, 6};
  for (int i = 0; i < 6; ++i) {
    switch (a[i]) {
    case 1:
      // CHECK: ONE
      printf("ONE\n");
      break;
    case 2:
      // CHECK: TWO
      printf("TWO\n");
      break;
    case 3:
      // CHECK: THREE
      printf("THREE\n");
      break;
    case 4:
      // CHECK: FOUR
      printf("FOUR\n");
    default:
      // CHECK: UNKNOWN
      // CHECK: UNKNOWN
      // CHECK: UNKNOWN
      printf("UNKNOWN\n");
    }
  }
  return 0;
}
