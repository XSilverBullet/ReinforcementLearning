import numpy

def build_state(features):
    return int("".join(map(lambda feature: str(int(feature)), features)))

#print value in place of bins
def to_bin(value, bins):
    return numpy.digitize(x=[value], bins=bins)[0]