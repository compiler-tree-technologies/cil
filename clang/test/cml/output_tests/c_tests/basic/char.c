// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  char ch = 'c';
  // CHECK: c
  printf("%c\n", ch);
  return 0;
}
