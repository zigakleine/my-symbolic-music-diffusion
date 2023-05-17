from magenta.models.music_vae import TrainedModel

import pickle

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.metrics import Metrics
from magenta.models.music_vae import TrainedModel
import note_seq
import config
import song_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')

# Model
flags.DEFINE_string('model', 'melody-2-big', 'Model configuration.')
flags.DEFINE_string('checkpoint', './musicvae_ckpt/cat-mel_2bar_big/cat-mel_2bar_big.ckpt.data-00000-of-00001',
                    'Model checkpoint.')

# Data transformation
flags.DEFINE_enum('mode', 'melody', ['melody', 'multitrack'],
                  'Data generation mode.')
flags.DEFINE_string('input',"./lakh_tfrecords", 'Path to tfrecord files.')
flags.DEFINE_string('output', "./out", 'Output path.')


logging.info('Loading pre-trained model %s', FLAGS.model)
model_config = config.MUSIC_VAE_CONFIG[FLAGS.model]
model = TrainedModel(model_config,
                          batch_size=1,
                          checkpoint_dir_or_path=FLAGS.checkpoint)

