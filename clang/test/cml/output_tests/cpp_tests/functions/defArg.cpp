// RUN: %cml -cpp %s -o %t && %t | FileCheck %s
#include<stdio.h>

void foo(int a, int b=10) {
   printf("SUM: %d\n",a+b);
}

int main() {
  foo(1); // CHECK: 11
  foo(2,7); // CHECK: 9
  return 0;
}
	
