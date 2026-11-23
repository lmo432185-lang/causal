"""Restore the declared zero-column matching axis lost in JSON's empty-list shape."""
import json
from core import unpack

def load_model(path):
 model=unpack(json.loads(path.read_text()));model['B_axis']=model['B_axis'].reshape(model['rank_B'],1);return model
