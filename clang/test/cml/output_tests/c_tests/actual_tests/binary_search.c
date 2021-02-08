// RUN: %cml %s -o %t && %t | FileCheck %s
// C program to implement recursive Binary Search 
#include <stdio.h> 
  
// A recursive binary search function. It returns 
// location of x in given array arr[l..r] is present, 
// otherwise -1 
int binarySearch(int arr[], int l, int r, int x) 
{ 
    if (r >= l) { 
        int mid = l + (r - l) / 2; 
  
        // If the element is present at the middle 
        // itself 
        if (arr[mid] == x) 
            return mid; 
  
        // If element is smaller than mid, then 
        // it can only be present in left subarray 
        if (arr[mid] > x) 
            return binarySearch(arr, l, mid - 1, x); 
  
        // Else the element can only be present 
        // in right subarray 
        return binarySearch(arr, mid + 1, r, x); 
    } 
  
    // We reach here when element is not 
    // present in array 
    return  (0 - 1); 
} 
  
int main(void) 
{ 
    int arr[5];
    arr[0] = 2;
    arr[1] = 3;
    arr[2] = 4;
    arr[3] = 10;
    arr[4] = 40;
    int n = 5;
    int x = 10; 
    int result = binarySearch(arr, 0, n - 1, x); 
    (result == (0 - 1) ? printf("Element is not present in array\n") 
                   : printf("Element is present at index %d\n",   //CHECK: present at index 3
                            result)); 
    x = 45; 
    result = binarySearch(arr, 0, n - 1, x); 
    (result == (0 - 1) ? printf("Element is not present in array\n") //CHECK: not present in array
                   : printf("Element is present at index %d\n",   
                            result)); 
    return 0; 
} 
