import logging
import models.rpvnet as rpvnet
import models.minkunet as minkunet
import models.segminkunet as segminkunet
import models.pointminkunet as pointminkunet
import models.mrfnet as mrfnet
import models.doubleminkunet as doubleminkunet


MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a or 'MLP' in a])


add_models(rpvnet)
add_models(minkunet)
add_models(segminkunet)
add_models(pointminkunet)
add_models(mrfnet)
add_models(doubleminkunet)


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = MODELS
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    logging.info(f'Invalid model index. You put {name}. Options are:')
    # Display a list of valid model names
    for model in all_models:
      logging.info('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass
