from word_suggestion import Word_Suggestion, Suggest_Util
import pydotplus

import tensorflow as tf
tf.enable_eager_execution()


model_metadata = Suggest_Util.load_dict("../config/model_metadata.json")
word_suggest = Word_Suggestion(*model_metadata.values())
word_suggest.build_model()
word_suggest.model =  tf.keras.models.load_model('../models/training_checkpoints/conversation_no_update/conv_model.h5')

tf.keras.utils.plot_model(
    word_suggest.model,
    to_file='model.png',
    show_shapes=True,
    show_layer_names=True,
    rankdir="LR"
)