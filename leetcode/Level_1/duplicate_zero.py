arr = [1,0,2,3,0,4,5,0]
class Solution:
    def duplicateZeros(self, arr: List[int]) -> None:
        """
        Do not return anything, modify arr in-place instead.
        """
        i = 0
        n = len(arr)
        while i < n:
            if arr[i] == 0:
                arr.pop()
                arr.insert(i+1, 0)
                i += 2
            else:
                i+=1