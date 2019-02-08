var gImage
var colorValue

function preload() {
	gImage = loadImage('frame_00006.png');
}

function setup() {
	createCanvas(windowWidth, windowHeight);
	background(100);
	
	const imageCanvas = createOpencvCanvas(0, 0);;
	loadPixels();
}

function draw() {
	clear()
	image(gImage, 0, 0);
	colorValue = get(mouseX,mouseY)
	fill(colorValue)
	square(gImage.width, 0, 55);
	fill(colorValue);
	text('x:'+mouseX, gImage.width+40, 100);
	fill(colorValue);
	text('y:'+mouseY, gImage.width+40, 130);
}

function mousePressed() {
  print('x:'+mouseX)
	print('y:'+mouseY)
	
  }


