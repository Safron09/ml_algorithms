nums = [-4,-1,0,3,10]
class Solution:
    def sortedSquares(self, nums: List[int]) -> List[int]:
        sorted = []
        for i in nums:
            sorted.append(i ** 2)
            sorted.sort()
        return sorted
            