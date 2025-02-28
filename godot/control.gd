extends Control

static func image_to_float_array(image: Image) -> Array[float]:
	var float_array: Array[float] = []
	
	for y in range(image.get_height()):
		for x in range(image.get_width()):
			var color = image.get_pixel(x, y)
			float_array.append(color.r * 0.2989 + color.g * 0.5870 + color.b * 0.1140)
	
	return float_array

func feed_network():
	var outputs: Array[float] = %NeuralNetwork.feed(image_to_float_array(%Canvas.image))
	for i in range(outputs.size()):
		%Probabilities.set_progress(i, outputs[i] * 100.0)

func _on_brush_size_spin_slider_value_changed(value: float) -> void:
	%Canvas.brush_size = value

func _on_canvas_brush_size_changed(value: int) -> void:
	%BrushSizeSpinSlider.value = float(value)

func _on_load_coefficients_dialog_file_selected(path: String) -> void:
	if not %NeuralNetwork.load_coefficients(path):
		%LoadCoefficientsDialog.visible = true

func _on_canvas_image_changed() -> void:
	feed_network()

func _on_neural_network_network_changed() -> void:
	feed_network()
