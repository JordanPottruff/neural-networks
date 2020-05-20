
var SIZE = 700;
var BACKGROUND_COLOR = 250;
var STROKE_COLOR = 230;


var penStrength;
var penRadius;
var penFade;

var grid = new Array(28*28);



function setup() {
    penStrength = document.getElementById("in-strength").value;
    penRadius = document.getElementById("in-radius").value;
    penFade = document.getElementById("in-fade").value;

    for(let i=0; i<grid.length; i++) {
        grid[i] = 0.0;
    }
    frameRate(30);
    let canvas = createCanvas(SIZE,SIZE);
    canvas.parent("canvas");
}

function draw() {
    background(250);
    drawGrid();
    if(mouseIsPressed) {
        addDrawing(mouseX, mouseY);
    }
}

function addDrawing(mx, my) {
    gx = gridX(mx);
    gy = gridY(my);
    addDrawingGrid(gx, gy, gx, gy, new Array());
}

function addDrawingGrid(gx, gy, cx, cy, visited) {
    if(gx > 27 || gx < 0 || gy > 27 || gy < 0) {
        return;
    }
    index = gx*28 + gy
    if(visited.includes(index)) {
        return;
    }
    let dist = Math.sqrt(Math.pow(gx-cx, 2) + Math.pow(gy-cy, 2));
    console.log(dist);
    if(dist >= penRadius) {
        return;
    }

    let increment = penStrength * Math.pow((penRadius - dist)/penRadius, penFade);
    grid[index] = Math.min(grid[index] + increment, 1.0);
    visited.push(index);
    addDrawingGrid(gx + 1, gy, cx, cy, visited);
    addDrawingGrid(gx - 1, gy, cx, cy, visited);
    addDrawingGrid(gx, gy + 1, cx, cy, visited);
    addDrawingGrid(gx, gy - 1, cx, cy, visited);
}

function gridX(mx) {
    return Math.floor(mx/(SIZE/28));
}

function gridY(my) {
    return Math.floor(my/(SIZE/28));
}

function drawGrid() {
    let squareSize = SIZE/28;
    for(let i=0; i<grid.length; i++) {
        let row = i%28;
        let col = Math.floor(i/28);

        let x = col*squareSize;
        let y = row*squareSize;
        strokeWeight(1.0);
        stroke(STROKE_COLOR);
        fill(255-(grid[i]*255));
        square(x, y, squareSize);
    }
}

function clearGrid() {
    for(let i=0; i<grid.length; i++) {
        grid[i] = 0.0;
    }
}

function changePenStrength(value) {
    penStrength = value;
}

function changePenRadius(value) {
    penRadius = value;
}

function changePenFade(value) {
    penFade = value;
}