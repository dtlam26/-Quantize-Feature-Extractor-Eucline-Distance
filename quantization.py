import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_frozen_graph(
                graph_def_file='eval_graph_frozen_124.pb',
                input_arrays=["input_image"],
                input_shapes= {"input_image" : [1,124,124,3]},
                output_arrays=['model/dense/act_quant/FakeQuantWithMinMaxVars'])
converter.quantized_input_stats = {"input_image" : (128, 127)}
converter.inference_type = tf.uint8
converter.inference_input_type = tf.uint8
quantized_tflite_model = converter.convert()
with open("test.tflite", 'wb') as f:
    f.write(quantized_tflite_model)
