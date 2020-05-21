
function classify(attributes) {
    console.log("classify");
    weights = RELU['weights']
    biases = RELU['biases']

    let output = attributes;
    for(let i=0; i<weights.length; i++) {
        weight = weights[i];
        bias = biases[i];
        if(i < weights.length - 1) {
            output = relu_func(add(multiply(weight, output), bias));
        } else {
            output = logistic_func(add(multiply(weight, output), bias));
        }

    }
    return output;
}

function add(vec1, vec2) {
    if(vec1.length != vec2.length) {
            console.log("add error");
            return NaN;
    }

    let sum = new Array(vec1.length).fill(0);
    for(let i=0; i<vec1.length; i++) {
        sum[i] = vec1[i] + vec2[i]
    }
    return sum;
}

function dot(vec1, vec2) {
    if(vec1.length != vec2.length) {
        console.log("dot error");
        return NaN;
    }

    let sum = 0;
    for(let i=0; i<vec1.length; i++) {
        sum += vec1[i] * vec2[i];
        if(!isNaN(vec1[i]) && !isNaN(vec2[i]) && isNaN(sum)){
            console.log(issue);
        }
    }
    return sum;
}

function multiply(mat, vec) {
    let result = new Array(mat[0].length).fill(0);
    for(let row=0; row<mat[0].length; row++) {
        for(let col=0; col<mat.length; col++) {
            old_result = Number(result[row]);
            result[row] += mat[col][row] * vec[col];
            if(!isNaN(mat[col][row]) && !isNaN(vec[col]) && isNaN(result[row])) {
                console.log("issue");
                console.log(mat[col][row]);
                console.log(vec[col]);
                console.log(old_result);
            }
        }
    }
    return result;
}

function transposeMat(mat) {
    let result = new Array(mat.length);
    for(let i=0; i<mat.length; i++) {
        result[i] = new Array(mat[0].length).fill(0);
    }
    for(let col=0; col<mat.length; col++) {
        for(let row=0; row<=col; row++) {
            result[col][row] = mat[row][col]
            result[row][col] = mat[col][row]
        }
    }
    return result;
}

function logistic_func(vec) {
    let result = new Array(vec.length).fill(0);
    for(let i=0; i<vec.length; i++) {
        result[i] = 1.0 / (1 + Math.exp(-vec[i]))
        if(isNaN(vec[i]) && !isNaN(result[i])) {
            console.log("issue");
        }
    }
    return result;
}

function relu_func(vec) {
    let result = new Array(vec.length).fill(0);
    for(let i=0; i<vec.length; i++) {
        result[i] = Math.max(0.0, vec[i]);
        if(isNaN(vec[i]) && !isNaN(result[i])) {
            console.log("issue");
        }
    }
    return result;
}