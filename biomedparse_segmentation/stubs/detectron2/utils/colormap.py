import random

def random_color(rgb=False, maximum=255):
    r = random.randint(0, maximum)
    g = random.randint(0, maximum)
    b = random.randint(0, maximum)
    return (r, g, b) if rgb else (b, g, r)
