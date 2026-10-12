# Practical 9 - Association Rule Mining (Apriori)
# Using Grocery Transactions
from itertools import combinations

# Each transaction = items bought together in one bill
transactions = [
    {"bread", "milk"},
    {"bread", "butter", "eggs"},
    {"milk", "butter", "eggs", "bread"},
    {"bread", "milk", "butter"},
    {"milk", "eggs"},
    {"bread", "butter"},
    {"bread", "milk", "eggs"},
    {"milk", "butter"}
]
n = len(transactions)
min_support = 0.4
min_confidence = 0.6

# Support = fraction of transactions that contain the itemset
def support(itemset):
    count = sum(1 for t in transactions if itemset <= t)
    return count / n

# Apriori: find frequent itemsets (size 1, then 2, then 3 ...)
items = sorted({item for t in transactions for item in t})
current = [frozenset([item]) for item in items]
frequent = {}
size = 1
while current:
    level = {c: support(c) for c in current if support(c) >= min_support}
    frequent.update(level)
    size += 1
    current = list({a | b for a in level for b in level if len(a | b) == size})

print("Frequent Itemsets:")
for itemset, sup in frequent.items():
    print(set(itemset), "Support:", round(sup, 2))

# Generate association rules with support and confidence
print("\nAssociation Rules:")
for itemset in frequent:
    for r in range(1, len(itemset)):
        for left in combinations(itemset, r):
            left = frozenset(left)
            right = itemset - left
            confidence = frequent[itemset] / frequent[left]
            if confidence >= min_confidence:
                print(set(left), "->", set(right),
                      "| Support:", round(frequent[itemset], 2),
                      "| Confidence:", round(confidence, 2))
