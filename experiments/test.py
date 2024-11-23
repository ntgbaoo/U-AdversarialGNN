degree_values = [(1, 3), (2, 1), (3, 5)]
min_degree = 2
valid_node = [node for (node, degree) in degree_values if degree > min_degree]
print(valid_node)  # Output: [1, 3]