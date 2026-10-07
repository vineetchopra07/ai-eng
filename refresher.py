x = [0,1,2,2,3,0,4,2]
val = 2
for i, v in enumerate(x):
    # print(i, v, "length before deletion: ", len(x))
    if v == val:
        print(f"Deleting element at index {i}: {v} -------------")
        del x[i]
        print("new current element:", x[i])
        # print(i, v, "length after deletion: ", len(x))
        # i=i-1

# print("Final list:", x)