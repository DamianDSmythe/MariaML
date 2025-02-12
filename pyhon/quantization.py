import tensorflow as tf

# Convert to TFLite with quantization
converter = tf.lite.TFLiteConverter.from_saved_model('Models', 'MariaMLV00002.h5')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the quantized model
with open('model_quantized.tflite', 'wb') as f:
    f.write(tflite_model)
