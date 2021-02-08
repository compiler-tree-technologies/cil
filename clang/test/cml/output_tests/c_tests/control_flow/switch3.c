// RUN: %cml %s -o %t && %t | FileCheck %s

#include <stdio.h>
int main() {
  int a = 1, b = 1;
  switch(a) {
    default: 
      b = 10;
      break;
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
  printf("%d\n",b); // CHECK: 3
  return 0;
}
