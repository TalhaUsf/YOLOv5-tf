from rich.console import Console

with open("utils/classes.txt", "r") as f:
    classes = f.readlines()

classes = [c.strip() for c in classes]

classes.pop(0) # remove the first element background class
Console().print(classes, justify="left", highlight=True)

class2idx = {c:idx for idx, c in enumerate(classes)}
Console().print(class2idx, justify="left", highlight=True)