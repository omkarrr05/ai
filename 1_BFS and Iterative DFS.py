# Practical 1 - BFS and Iterative DFS
# Sindhudurg Talukas

graph = {
    "Sawantwadi": ["Vengurla", "Kudal"],
    "Vengurla": ["Sawantwadi", "Kudal"],
    "Kudal": ["Vengurla", "Sawantwadi", "Malvan", "Kankavli"],
    "Malvan": ["Kudal", "Devgad"],
    "Kankavli": ["Kudal", "Devgad"],
    "Devgad": ["Kankavli", "Malvan"],
}


# BFS - Queue
def bfs(start, goal):
    visited = []
    queue = [start]

    while queue:
        place = queue.pop(0)

        if place not in visited:
            visited.append(place)

            if place == goal:
                return visited

            queue.extend(graph[place])

    return visited


# DFS - Stack
def dfs(start, goal):
    visited = []
    stack = [start]

    while stack:
        place = stack.pop()

        if place not in visited:
            visited.append(place)

            if place == goal:
                return visited

            stack.extend(graph[place])

    return visited


start = "Vengurla"
goal = "Kankavli"

print("Start Taluka:", start)
print("Goal Taluka:", goal)

print("BFS:", bfs(start, goal))
print("DFS:", dfs(start, goal))
