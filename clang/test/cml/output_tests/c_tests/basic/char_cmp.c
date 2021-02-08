// RUN: %cml %s -o %t && %t | FileCheck %s
#include <stdio.h>
int main() {
  char ch = 'c';
  // CHECK: c
  printf("%c\n", ch);
  if (ch == 'c')
    // CHECK: Char is c
    printf("Char is c\n");
  return 0;
}
