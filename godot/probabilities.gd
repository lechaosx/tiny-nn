@tool

extends GridContainer

@export var values: Array[String] = []

func _ready() -> void:
	for value in values:
		var progress_bar = ProgressBar.new()
		var label = Label.new()
		
		progress_bar.fill_mode = ProgressBar.FILL_END_TO_BEGIN
		progress_bar.show_percentage = false
		progress_bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		progress_bar.size_flags_vertical = Control.SIZE_FILL
		label.text = value
		
		add_child(progress_bar)
		add_child(label)
