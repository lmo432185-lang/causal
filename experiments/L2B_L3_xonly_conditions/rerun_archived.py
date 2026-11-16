"""Use archived inputs, without the original workspace copy paths. Run in a copy."""
from scan import run as scan
from structure import run as structure
from equivalence import run as equivalence
if __name__=='__main__':scan();structure();equivalence()
