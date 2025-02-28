extends TextureRect

signal brush_size_changed(value: int)
signal image_changed()

@export var image_size: Vector2i = Vector2i(28, 28)

var image: Image

var brush_size: int = 1:
	get: return brush_size
	set(new_brush_size):
		if brush_size != new_brush_size:
			brush_size = new_brush_size
			brush_size_changed.emit(new_brush_size)
			
func _ready() -> void:
	image = Image.create(image_size.x, image_size.y, false, Image.FORMAT_RGB8)
	image.fill(Color(0, 0, 0))
	
	custom_minimum_size = image_size
	
	texture = ImageTexture.create_from_image(image)

func _process(_delta: float) -> void:
	var drawing_mode = int(Input.is_mouse_button_pressed(MOUSE_BUTTON_LEFT)) - int(Input.is_mouse_button_pressed(MOUSE_BUTTON_RIGHT))

	if drawing_mode != 0:
		var bounding_rectangle = Rect2(Vector2(0, 0), image_size)
		
		var draw_position = get_local_mouse_position() * Vector2(image_size) / size
		if bounding_rectangle.has_point(draw_position):
			var color = Color.WHITE if drawing_mode > 0 else Color.BLACK
			draw_brush(draw_position, color)
			texture.update(image)
			image_changed.emit()
			
	queue_redraw()

func _gui_input(event: InputEvent) -> void:
	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_WHEEL_UP and event.pressed:
			brush_size = min(brush_size + 1, 20)
		if event.button_index == MOUSE_BUTTON_WHEEL_DOWN and event.pressed:
			brush_size = max(brush_size - 1, 1)

func _draw() -> void:
	draw_circle(get_local_mouse_position(), brush_size * (size.x / image.get_size().x), Color(1, 1, 1, 0.5), false, 0.5, true)

func draw_brush(pos: Vector2, color: Color):
	for y in range(image.get_height()):
		for x in range(image.get_width()):
			var intensity = gaussian_brush(pos.distance_to(Vector2(x, y) + Vector2(0.5, 0.5)))
			
			var pixel = image.get_pixel(x, y)
			pixel = lerp(pixel, color, intensity)
			image.set_pixel(x, y, pixel)

func gaussian_brush(distance: float):
	return exp(-pow(distance, 2) / (2 * pow(brush_size / 3.0, 2)))
