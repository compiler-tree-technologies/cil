//RUN: %cml -emit-ir -cpp %s -o -| FileCheck %s -check-prefix=OPTNONE
//RUN: %cml -optStl -emit-ir -cpp %s -o -| FileCheck %s -check-prefix=OPT
#include<stdio.h>
#include<vector>

int main() {
 std::vector<int> vec;
 std::vector<int> vec2;


 for(int i=0; i<10; i++) {
   vec.push_back(i);
 }

 vec2.reserve(10);
 for(int i=0; i<10; i++) {
   vec2.push_back(i+2);
 }
 
 printf("%d %d",vec[2],vec[7]);
 printf("%d %d",vec2[3],vec2[6]);

 return 0;
}

//OPTNONE:_ZNSt3__16vectorIiNS_9allocatorIiEEE7reserveEm

//OPT:_ZNSt3__16vectorIiNS_9allocatorIiEEE7reserveEm
//OPT:_ZNSt3__16vectorIiNS_9allocatorIiEEE7reserveEm
