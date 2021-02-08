// RUN: %cml %s -o %t && %t | FileCheck %s

#include <stdio.h>
int main() {
  int a = 2, b = 1;
  switch(a) {
    default: 
      b = 10;
      break;
    case 1: {
      b = 3;
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
  printf("%d\n",b); // CHECK: 5
  return 0;
}
