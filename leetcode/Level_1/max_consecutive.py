nums = [1,1,0,1,1,1]
class Solution:
    def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
        streak = 0
        count = 0
        for i in nums:
            if i == 1:
                count += 1
                streak = max(streak, count)
            else:
                count = 0
        return streak
        
