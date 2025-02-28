extends VBoxContainer

signal min_value_changed(value: float)
signal max_value_changed(value: float)
signal value_changed(value: float)

@export var text: String = ""

@export var min_value: float = 0.0:
	get: return min_value
	set(new_min_value):
		if min_value != new_min_value:
			min_value = new_min_value
			min_value_changed.emit(new_min_value)
	
@export var max_value: float = 100.0:
	get: return max_value
	set(new_max_value):
		if max_value != new_max_value:
			max_value = new_max_value
			max_value_changed.emit(new_max_value)
			
@export var value: float = 0.0:
	get: return value
	set(new_value):
		if value != new_value:
			value = new_value
			value_changed.emit(new_value)

func _ready():
	%Label.text = text
	
	%SpinBox.min_value = min_value
	%SpinBox.max_value = max_value
	%SpinBox.value = value
	
	%HSlider.min_value = min_value
	%HSlider.max_value = max_value
	%HSlider.value = value
	
func _on_spin_box_value_changed(new_value: float) -> void:
	value = new_value

func _on_h_slider_value_changed(new_value: float) -> void:
	value = new_value

func _on_max_value_changed(new_value: float) -> void:
	%SpinBox.max_value = new_value
	%HSlider.max_value = new_value

func _on_min_value_changed(new_value: float) -> void:
	%SpinBox.min_value = new_value
	%HSlider.min_value = new_value
	
func _on_value_changed(new_value: float) -> void:
	%SpinBox.value = new_value
	%HSlider.value = new_value
