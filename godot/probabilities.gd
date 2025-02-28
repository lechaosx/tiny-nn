@tool

extends GridContainer

@export var values: Array[String] = []

var _progress_bars: Array[ProgressBar] = []

func _ready() -> void:
	for value in values:
		var progress_bar = ProgressBar.new()
		var label = Label.new()
		
		progress_bar.fill_mode = ProgressBar.FILL_END_TO_BEGIN
		progress_bar.show_percentage = false
		progress_bar.size_flags_horizontal = Control.SIZE_EXPAND_FILL
		progress_bar.size_flags_vertical = Control.SIZE_FILL
		label.text = value
		
		_progress_bars.append(progress_bar)
		
		add_child(progress_bar)
		add_child(label)

func set_progress(index: int, value: float) -> void:
	_progress_bars[index].value = clamp(value, 0, 100)
