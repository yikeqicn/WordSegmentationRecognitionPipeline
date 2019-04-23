import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = './frozen_east_text_detection.pb'

with tf.Session() as sess:
   print("load graph")
   with tf.gfile.GFile(GRAPH_PB_PATH,'rb') as f:
       graph_def = tf.GraphDef()
   graph_def.ParseFromString(f.read())
   sess.graph.as_default()
   # name="" is important to ensure we don't get spurious prefixing
   tf.import_graph_def(graph_def, name='')
   graph_nodes=[n for n in graph_def.node]
   names = []
   for t in graph_nodes:
      names.append(t.name)
   print(names)
   g = tf.get_default_graph()
   inp = g.get_tensor_by_name("input_images:0")
   out = g.get_tensor_by_name("feature_fusion/Conv_7/Sigmoid:0")
   geometry = g.get_tensor_by_name("feature_fusion/concat_3:0")
   tf.compat.v1.saved_model.simple_save(
      sess,
      "./2/",
      inputs={"image": inp},
      outputs={"scores": out, "geometry": geometry},
      legacy_init_op=tf.tables_initializer(),
   )

   """
   with tf.Session(graph=tf.Graph()) as sess:
      
      tf.import_graph_def(graph_def, name="")
      g = tf.get_default_graph()
      inp = g.get_tensor_by_name("real_A_and_B_images:0")
      out = g.get_tensor_by_name("generator/Tanh:0")

      sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
         tf.saved_model.signature_def_utils.predict_signature_def(
               {"in": inp}, {"out": out})

      builder.add_meta_graph_and_variables(sess,
                                          [tag_constants.SERVING],
                                          signature_def_map=sigs)

   builder.save()

   tf.compat.v1.saved_model.simple_save(
         sess,
         "./1/",
         inputs={"text": graph_def.input_images},
         outputs={"prediction": graph_def.outputs[0]},
         legacy_init_op=tf.tables_initializer(),
   )
   """