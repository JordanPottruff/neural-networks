
var SIZE = 700;
var BACKGROUND_COLOR = 250;
var STROKE_COLOR = 230;
var PROB_MAX_BRIGHTNESS = 64;
var UPDATE_DELAY = 250; // Milliseconds.


var penStrength;
var penRadius;
var penFade;

var grid = new Array(28*28);

var popupOpen = false;
var detailsOpen = false;
var realTimeOn = false;

var lastGuessTime = new Date().getTime();
var update = false;

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

    document.getElementById("canvas").addEventListener("touchstart", function(event) {event.preventDefault()});
    document.getElementById("canvas").addEventListener("touchmove", function(event) {event.preventDefault()});
    document.getElementById("canvas").addEventListener("touchend", function(event) {event.preventDefault()});
    document.getElementById("canvas").addEventListener("touchcancel", function(event) {event.preventDefault()});

    let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
}

function draw() {
    background(250);
    drawGrid();
    if(realTimeOn && new Date().getTime() - lastGuessTime > UPDATE_DELAY && update) {
        updateGuess();
        lastGuessTime = new Date().getTime();
        update = false;
    }
    if(mouseIsPressed && !popupOpen) {
        update = true;
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
    index = gy*28 + gx;
    if(visited.includes(index)) {
        return;
    }
    let dist = Math.sqrt(Math.pow(gx-cx, 2) + Math.pow(gy-cy, 2));
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
        let col = i%28;
        let row = Math.floor(i/28);

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
    updateGuess();
}

function centerGrid() {
    let lowCol = 28;
    let highCol = 0;
    let lowRow = 28;
    let highRow = 0;


    for(let i=0; i<grid.length; i++) {
        let col = i%28;
        let row = Math.floor(i/28);

        if(grid[i] > 0.0001) {
            lowCol = Math.min(lowCol, col);
            highCol = Math.max(highCol, col);
            lowRow = Math.min(lowRow, row);
            highRow = Math.max(highRow, row);
        }
    }

    let shiftDown = Math.floor((28 - (highRow - lowRow)) / 2) - (27-highRow);
    let shiftLeft = Math.floor((28 - (highCol - lowCol)) / 2) - (27-highCol);

    let newGrid = new Array(28*28).fill(0);
    for(let i=0; i<grid.length; i++) {
        let col = i%28;
        let row = Math.floor(i/28);

        let shiftedCol = Math.max(col - shiftLeft, 0);
        let shiftedRow = Math.max(row - shiftDown, 0);

        let shiftedI = shiftedRow * 28 + shiftedCol;
        if(shiftedI < 28*28 && shiftedI > 0) {
            newGrid[shiftedI] = grid[i];
        }
    }
    return newGrid;
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

function submit() {
    updateGuess();
    document.getElementById("overlay").style.display = "flex";

    popupOpen = true;
}

function updateGuess() {
    output = classify(centerGrid(grid));
    document.getElementById("popup-guess").innerHTML = getGuess(output);
    orderedOutput = orderGuesses(output);
    let table = document.getElementById("prob-table");
    for(let i=0; i<orderedOutput.length; i++) {
        let dig = orderedOutput[i]["dig"];
        let prob = orderedOutput[i]["prob"] * 100;
        let brightness = 255 - (PROB_MAX_BRIGHTNESS * (prob/100));
        console.log(brightness);

        let tableRow = document.getElementById("guess" + i);
        tableRow.children[0].innerHTML = dig;
        tableRow.style.backgroundColor = "rgb("+brightness+",255,"+brightness+")";
        tableRow.children[1].innerHTML = prob.toFixed(2) + "%";
    }
}

function closePopup() {
    document.getElementById("overlay").style.display = "none";
    popupOpen = false;
}

function toggleDetails() {
    if(detailsOpen) {
        // Close more details
        document.getElementById("probabilities").style.display = "none";
        document.getElementById("details").innerHTML = "Show";
    } else {
        // Open more details
        document.getElementById("probabilities").style.display = "flex";
        document.getElementById("details").innerHTML = "Hide";
    }
    detailsOpen = !detailsOpen;
}

function getGuess(output) {
    let maxIndex = 0;
    for(let i=1; i<output.length; i++) {
        if(output[i] > output[maxIndex]) {
            maxIndex = i;
        }
    }
    return maxIndex;
}

function orderGuesses(output) {
    let map = new Array(output.length);
    for(let i=0; i<output.length; i++) {
        map[i] = {"dig":i, "prob":output[i]};
    }

    map.sort((a, b) => (a["prob"] < b["prob"]) ? 1 : -1);
    return map;
}

function toggleRealTime() {
    let buttonSubmit = document.getElementById('button-submit');
    let overlay = document.getElementById('overlay');
    let submissionPopup = document.getElementById('submission-popup');
    let rightSidebar = document.getElementById('right-sidebar');
    let buttonClosePopup = document.getElementById('popup-close');

    if(realTimeOn) {
        // Turn off real-time.
        buttonSubmit.disabled = false;
        overlay.appendChild(submissionPopup);
        rightSidebar.style.display = "none";
        buttonClosePopup.style.display = "inline-block";
        // Revert details option.
        toggleDetails();
        toggleDetails();
        document.getElementById("popup-more").style.display = "block";
    } else {
        // Turn on real-time.
        buttonSubmit.disabled = true;
        rightSidebar.appendChild(submissionPopup);
        rightSidebar.style.display = "block";
        buttonClosePopup.style.display = "none";
        updateGuess();
        // Show details in real-time mode.
        document.getElementById("probabilities").style.display = "flex";
        document.getElementById("details").innerHTML = "Hide";
        document.getElementById("popup-more").style.display = "none";
    }
    realTimeOn = !realTimeOn;
}