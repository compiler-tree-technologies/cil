// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>

int main() {
  int a = 1;
  switch (a) {
  case 1:
  case 2:
  case 3:
    // CHECK: ONETWOTHREE
    printf("ONETWOTHREE\n");
    break;
  case 4:
    printf("FOUR\n");
    break;
  default:
    printf("UNKNOWN\n");
  }
  return 0;
}
