extends Control

func _on_brush_size_spin_slider_value_changed(value: float) -> void:
	%Canvas.brush_size = value


func _on_canvas_brush_size_changed(value: int) -> void:
	%BrushSizeSpinSlider.value = float(value)
