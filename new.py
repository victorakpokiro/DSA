



def main(s, t):

	#return sorted(s) ==sorted(t)   #big O of 1

	if len(s) != len(t):
		print("failed")
		return False

	countS, countT = {}, {}

	for i in range(len(s)):
		countS[s[i]] = 1 + countS.get(s[i], 0)
		countT[t[i]] = 1 + countT.get(t[i], 0)
	print(countS, len(countS))
	print(countT, len(countT))

	# for i in countS:
	# 	if countS[i] != countT.get(i, 0):
	# 		print("failed2")
	# 		return False

	# return True
	for i in countS:
		if countS != countT:
			print("failed2")
			return False

	return True		


def main2(arr, target):
	save = {} #value : index

	for index, num in enumerate(arr):
		diff = target - num 
		if diff in save:
			print([index, save[diff]])
		save[num] = index


def main3(num):
	large_sum = num[0]
	curr_sum = 0

	for i in num:
		curr_sum += i
		if curr_sum < 0:
			curr_sum = 0
		large_sum = max(large_sum, curr_sum)
	print(large_sum)
	return large_sum


# array has to be sorted O(nlogn)
def main4(arr, target):
	#using 2 pointers with big O of n
	l, r = 0, len(arr)-1

	while l < r:
		curr_sum = arr[l] + arr[r]

		if curr_sum > target:
			r -= 1
		elif curr_sum < target:
			l +=1
		else:
			print([l+1, r+1])
			return
	return [l+1, r+1]



#using 2-pointer to determine the difference to equal the target
def main6(arr, target):
	l, r = 0, 1

	while l < r:
		curr_diff = arr[r] - arr[l]

		if curr_diff < target :
			r += 1
		elif curr_diff > target:
			l += 1
		else:
			return ([arr[l], arr[r]])



def rob(arr):
	rob1, rob2 = 0, 0

	#[rob1, rob2, n, n+1, ...]
	for i in arr:
		temp = max(i+rob1, rob2)
		rob1 = rob2
		rob2 = temp
	return rob2



#LinkedList
class ListNode:
	"""docstring for ListNode"""
	def __init__(self, val=0, next=None):
		self.val = val 
		self.next = next

#l1 and l2 are listnode objects
def add_substring(l1, l2):
	node = ListNode()
	tail = node

	while l1 and l2:
		if l1.val < l2.val:
			tail.next = l1
			l1 = l1.next 
		else:
			tail.next = l2
			l2 = l2.next
		tail = tail.next

	if l1:
		tail.next = l1 
	elif l2:
		tail.next = l2

	return node.next


#2-pointer profit problem
def main5(price):
	l, r = 0, 1
	maxP = 0

	while r < len(price):
		if price[l] < price[r]:
			profit = price[r] - price[l]
			maxP = max(profit, maxP)
		else:
			l=r
		r +=1
	return maxP


def slidewindow(arr, target):
	n = len(arr)
	if n < target:
		print("invalid")
		return False

	curr_sum = 0
	for i in range(target):
		curr_sum += arr[i]

	maxm = curr_sum
	for i in range(target, n):
		curr_sum += arr[i] - arr[i-target]
		maxm = max(curr_sum, maxm)
	print(maxm)


#k = target
def slidewindow2(arr, k):
	n = len(arr)
	if n < k:
		print("invalid")
		return False

	res = []
	first_max = res.append(max(arr[0:k]))

	for i in range(k,n):
		res.append(max(arr[i-k+1: i+1]))
	print(res)


#smallest subarray sum greater or equal to 3
import math
def slidewindow3(arr, k):
	n, init = len(arr), arr[0]

	# if n < k:
	# 	print("invalid")
	# 	return False

	# res = [0]
	# curr_sum = init

	# for i in range(1,n):
	# 	curr_sum += arr[i]
	# 	res.append(i)
	# 	if curr_sum >= k:
	# 		print(res)
	# 		return True

	min_window_size = math.inf
	window_start = 0
	curr_sum = 0

	for i in range(n):
		curr_sum += arr[i]

		while curr_sum >= k:
			min_window_size = min(min_window_size, i-window_start+1)
			curr_sum -= arr[window_start]
			window_start += 1
	print(min_window_size) 


#longest substring with k distinct character
#using 2-pointers todetect a cycle in a linkedlist

def isValid(input):
	stack = []
	brackets = {"(":")", "[":"]", "{":"}"}

	for i in input:
		if i in brackets:
			if stack and stack[-1] == brackets[i]:
				stack.pop()
			else:
				return False
		else:
			stack.append(i)
	return len(stack) == 0

   leftSymbols = []
	# Loop for each character of the string
	# for c in input:
	#     # If left symbol is encountered
	#     if c in ['(', '{', '[']:
	#         leftSymbols.append(c)
	#     # If right symbol is encountered
	#     elif c == ')' and len(leftSymbols) != 0 and leftSymbols[-1] == '(':
	#         leftSymbols.pop()
	#     elif c == '}' and len(leftSymbols) != 0 and leftSymbols[-1] == '{':
	#         leftSymbols.pop()
	#     elif c == ']' and len(leftSymbols) != 0 and leftSymbols[-1] == '[':
	#         leftSymbols.pop()
	#     # If none of the valid symbols is encountered
	#     else:
	#         return False
	# return leftSymbols == []



#Given a string s, find the length of the longest substring without repeating characters.
class Solution:
	def lengthOfLongestSubstring(self, s: str) -> int:
		res = []
		m = 0
		
		for i in s:
			if i not in res:
				res.append(i)
				if len(res) > m:
					m = len(res)
			else:
				del res[: res.index(i)+1]
				res.append(i)
		return m


#Longest Palindromic Substring
#palindrome = l>=0 and r<len(s) and s[l]==s[r]
class Solution:
	def longestPalindrome(self, s: str) -> str:
		longest_len=0
		longest_str=''
		for i in range(len(s)):
			#checks for odd number length
			l,r=i,i
			while l>=0 and r<len(s) and s[l]==s[r]:
				if (r-l+1)>longest_len:
					longest_str=s[l:r+1]
					longest_len=r-l+1

				l-=1
				r+=1

			l,r=i,i+1
			#check for even number length
			while l>=0 and r<len(s) and s[l]==s[r]:
				if (r-l+1)>longest_len:
					longest_str=s[l:r+1]
					longest_len=r-l+1
				l-=1
				r+=1
		return longest_str


#Area = (r-l) * min(array[l], array[r])
#Container With Most Water
class Solution:
	def maxArea(self, height: List[int]) -> int:
		n = len(height) - 1
		l, r = 0, n
		water = 0
		
		while l < r:
			area = (r-l) * min(height[l], height[r])
			
			if height[l] < height[r]:
				l += 1
			else:
				r -= 1
				
			water = max(water, area)
		return water

#3 SUM
class Solution:
	def threeSum(self, nums: List[int]) -> List[List[int]]:
		#we can use a set so that we make sure that no duplicates are added
		result = set()
		n = len(nums)
		nums.sort()
		
		for i in range(n):
			l, r = i+1, n-1
			while l<r:
				curr_sum = nums[i] + nums[l] + nums[r]
				if curr_sum == 0:
					result.add((nums[i] , nums[l] , nums[r]))
					l += 1
					r -= 1
				elif curr_sum > 0:
					r -= 1
				else:
					l += 1
		return result


#Merge Two Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
	def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
		curr = node = ListNode()
		
		while (list1 and list2):
			if list1.val < list2.val:
				curr.next = list1
				list1, curr = list1.next, list1
			else:
				curr.next = list2
				list2, curr = list2.next, list2
		if list1 or list2:
			curr.next = list1 if list else list2
		return node.next


#Question number 9
#binary search
class Solution:
	def search(self, nums: List[int], target: int) -> int:
		
		start = 0
		end = len(nums) - 1
		while start <= end:
			mid = start + (end - start)// 2
			if nums[mid] == target: return mid
			if nums[mid] < target: start = mid + 1
			else: end = mid - 1
		return -1

#First Bad Version
class Solution:
	def firstBadVersion(self, n: int) -> int:
		
		left  = 1
		right = n


		while left < right:
			mid = (right + left) // 2
			if isBadVersion(mid) is False: 
				left = mid + 1 # if mid is false means, we have the bad version [true] on the right side

			else:
				right = mid # else we have bad version [true] on the left side
		return left 

#First Bad Version
class Solution:
	def firstBadVersion(self, n: int) -> int:
		left = 1
		right = n
		
		while right >= left:
			mid = (left+right) // 2
			if not isBadVersion(mid):
				left = mid + 1
			else:
				right = mid -1

		return left

#binary search with where missing value is
class Solution:
	def searchInsert(self, nums: List[int], target: int) -> int:
		start = 0
		end = len(nums) - 1
		while start <= end:
			mid = start + (end - start)// 2
			if nums[mid] == target: return mid
			if nums[mid] < target: start = mid + 1
			else: end = mid - 1
		return start


#Squares of a Sorted Array
class Solution:
	def sortedSquares(self, nums: List[int]) -> List[int]:
		res = [0] * len(nums)
		left = 0
		right = len(nums) - 1
		while left <= right:
			
			if abs(nums[left]) >= abs(nums[right]): 
				res[right - left] = nums[left] ** 2
				left += 1
			else: 
				res[right - left] = nums[right] ** 2
				right -= 1
		return res


#Rotate Array
class Solution:
	def rotate(self, nums: List[int], k: int) -> None:
		"""
		Do not return anything, modify nums in-place instead.
		"""

		def reverse(start, end):
			while start <= end:
				nums[start], nums[end] = nums[end], nums[start]
				start += 1
				end -= 1
		l = len(nums)
		k = k % l
		reverse(0, l-1 )
		reverse(0,k-1)
		reverse(k,l-1)


#Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
class Solution:
	def moveZeroes(self, nums: List[int]) -> None:
		"""
		Do not return anything, modify nums in-place instead.
		"""
		
		left = 0
		
		for i in range(len(nums)):
			if nums[i] != 0:
				nums[left], nums[i] = nums[i], nums[left]
				left += 1


#Write a function that reverses a string. The input string is given as an array of characters s.
class Solution:
	def reverseString(self, s: List[str]) -> None:
		"""
		Do not return anything, modify s in-place instead.
		"""
		
		left = 0
		right = len(s)-1
		
		while left <= right:
			s[left], s[right] = s[right], s[left]
			left += 1
			right -= 1


#Given a string s, reverse the order of characters in each word within a sentence while still preserving whitespace and initial word order.
class Solution:
	def reverseWords(self, s: str) -> str:
		a = s.split(" ")
		
		def reverse(n_s):
			s = list(n_s)
			l, r = 0, len(s)-1
			while l<r:
				s[l], s[r] = s[r], s[l]
				l += 1
				r -= 1
			return ''.join(s)
			
		for i in range(len(a)):
			a[i] = reverse(a[i])
			
		return ' '.join(a)


#Given the head of a singly linked list, return the middle node of the linked list.
class Solution:
	def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
		
		arr = []
		
		while head:
			arr.append(head)
			head = head.next
			
		mid = len(arr)//2
		return arr[mid]


#Given the head of a singly linked list, return the middle node of the linked list.
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
	def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
		
		slow, fast = head, head
		
		while fast and fast.next:
			slow = slow.next
			fast = fast.next.next
		return slow


#Given the head of a singly linked list, return the middle node of the linked list.
class Solution:
	def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
		
		curr = head
		count = 0
		
		while curr:
			curr = curr.next
			count += 1
			
		mid = count//2
		count = 0
		curr = head
		while count < mid and curr:
			curr = curr.next
			count += 1
		return curr


#Given the head of a linked list, remove the nth node from the end of the list and return its head.
class Solution:
	def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
		
		curr = head
		count = 0
		
		while curr:
			curr = curr.next
			count += 1
		count -= n
		curr = head
		prev = None
		while count > 0:
			prev = curr
			curr = curr.next
			count -= 1
			
		if prev:
			prev.next = curr.next
		else:
			head = head.next
		return head

#Given two strings s1 and s2, return true if s2 contains a permutation of s1, or false otherwise.
#In other words, return true if one of s1's permutations is the substring of s2.
class Solution:
	def checkInclusion(self, s1: str, s2: str) -> bool:
		def check_all_zero(arr):
			for num in arr:
				if num != 0:
					return False
			return True
		l1, l2 = len(s1), len(s2)
		if l1 > l2: return False
		arr = [0]*26
		for i in range(l1):
			arr[ord(s1[i]) - ord('a')] += 1
			arr[ord(s2[i]) - ord('a')] -= 1
		if check_all_zero(arr): return True
		
		for i in range(l1, l2):
			arr[ord(s2[i-l1]) - ord('a')] += 1
			arr[ord(s2[i]) - ord('a')] -= 1
			if check_all_zero(arr): return True
			
		return False


'''
An image is represented by an m x n integer grid image where image[i][j] represents the pixel value of the image.

You are also given three integers sr, sc, and newColor. You should perform a flood fill on the image starting from the pixel image[sr][sc].

To perform a flood fill, consider the starting pixel, plus any pixels connected 4-directionally to the starting pixel of the same color as the starting pixel, plus any pixels connected 4-directionally to those pixels (also with the same color), and so on. Replace the color of all of the aforementioned pixels with newColor.

Return the modified image after performing the flood fill.
'''
class Solution:
	def floodFill(self, image: List[List[int]], sr: int, sc: int, newColor: int) -> List[List[int]]:
		if image[sr][sc] == newColor: return image
		row, col = len(image), len(image[0])
		def dfs(r,c,color,newColor):
			if r not in range(row) or c not in range(col) or image[r][c] != color: return
			image[r][c] = newColor
			dfs(r+1, c, color, newColor)
			dfs(r-1, c, color, newColor)
			dfs(r, c+1, color, newColor)
			dfs(r, c-1, color, newColor)
			
		dfs(sr, sc, image[sr][sc], newColor)
		return image

'''
You are given an m x n binary matrix grid. An island is a group of 1's (representing land) connected 4-directionally (horizontal or vertical.) 
You may assume all four edges of the grid are surrounded by water.

The area of an island is the number of cells with a value 1 in the island.

Return the maximum area of an island in grid. If there is no island, return 0.
'''

class Solution:
	def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
		def dfs(grid, i, j):
			
			row_len = len(grid)
			col_len = len(grid[0])
			
			if i not in range(row_len) or j not in range(col_len) or grid[i][j] == 0: return 0
			
			grid[i][j] = 0
			count = 1
			count += dfs(grid, i+1, j)
			count += dfs(grid, i-1, j)
			count += dfs(grid, i, j+1)
			count += dfs(grid, i, j-1)
			return count
		
		maxm = 0
		
		for i in range(len(grid)):
			for j in range(len(grid[i])):
				if grid[i][j] == 1:
					maxm = max(maxm, dfs(grid, i, j))
		return maxm
		
		
'''
You are given two binary trees root1 and root2.

Imagine that when you put one of them to cover the other, some nodes of the two trees are overlapped while the others are not. 
You need to merge the two trees into a new binary tree. The merge rule is that if two nodes overlap, 
then sum node values up as the new value of the merged node. Otherwise, the NOT null node will be used as the node of the new tree.

Return the merged tree.

Note: The merging process must start from the root nodes of both trees.
'''

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
	def mergeTrees(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> Optional[TreeNode]:
		if root1 == None:
			return root2
		if root2 == None:
			return root1
		if root1 == None or root2 == None:
			return None
		
		root1.val += root2.val
		
		root1.left = self.mergeTrees(root1.left, root2.left)
		root1.right = self.mergeTrees(root1.right, root2.right)
		
		return root1


#Given the root of a binary tree, return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
class Solution:
	def levelOrder(self, root):
		result = []
		if root != None:
			self.traverse(root, 0, result)
		return result
	
	def traverse(self, node, row, result):
		if node != None:
			if len(result) < row+1:
				result.append([])
			result[row].append(node.val)
			self.traverse(node.left, row+1, result)
			self.traverse(node.right, row+1, result)


class Solution:

	def levelOrder(self, root: TreeNode) -> List[List[int]]:
		if root==None:
			return root
		def breadth_first_search(root, result, level):
			if root==None:
				return root
			if len(result) == level:
				result.append([])
			result[level].append(root.val)
			breadth_first_search(root.left,  result, level + 1)
			breadth_first_search(root.right, result, level + 1)

		result = []
		breadth_first_search(root, result, 0)
		return result











if __name__ == '__main__':
	# main("anagram", "naagrma")
	# main2([1,2,7,9], 9)
	# main3([1,2,-7,-9,2,3])
	# main4([1, 3, 4, 5, 7, 10], 8)
	# slidewindow([1, 4, 2, 10, 2, 3, 1, 0, 20], 4)
	slidewindow2([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 3)
	# slidewindow3([4,2,2,7,8,1,2,8,1,0], 8)
	# slidewindow4(["a", "a", "a", "b", "b", "m", "k"])