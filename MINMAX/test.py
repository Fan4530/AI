

def twoSum(self, nums, target):
    map = {};
    for i in range(0, len(nums)):
        if target - nums[i]  not in map:
            map[nums[i]] = i;
        else:
            return map[target - nums[i]] + 1, i + 1
    return -1, -1



# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

def addTwoNumbers(l1, l2):
    carry = 0
    root = n = ListNode(0);
    while l1 or l2 or carry:
        sum = carry;
        if l1:
            sum = sum + l1.val;
            l1 = l1.next;
        if l2:
            sum = sum + l2.val;
            l2 = l2.next;
        n.next = ListNode(sum % 10)
        carry = sum / 10;
        n = n.next;
    return root.next


l1 = ListNode(9)
l1.next = ListNode(4)
l2 = ListNode(9)


l = addTwoNumbers(l1, l2);
print l.val;
print l.next.val


print twoSum(1, [1,2,3,4], 5)


def lengthOfLongestSubstring(self, s):
    """
    :type s: str
    :rtype: int
    """
    #  wke
    #
    #  a b c a b c b b
    #    i   j
    # move j, when hash[j] != 0, move j
    # globalMax
    # key: number   value: index
    res = 0;
    left = 0;

    hash = {};
    for i in range(0, len(s)):
        if s[i] in hash and left <= hash[s[i]] :
            left = hash[s[i]] + 1
        else:
            res = max(res, i - left + 1)
        print i, left
        hash[s[i]] = i
    return res

