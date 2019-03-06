const PROCESSING_RESOLUTION_WIDTH = 480;
const FPS = 30;

let width = 0;
let height = 0;
let streaming = false;

let video = document.querySelector("#videoInput");
let tempCanvas = document.querySelector("#tempCanvas");
let outputCanvas = document.querySelector("#outputCanvas");
let startAndStopButton = document.querySelector('#startAndStop');
let colorButton = document.querySelector('#color');
let usingGray = false;
let detactFeatures = false;

let keyPoints1;
let mask;

var Module = {
    setStatus: function(text) {
        console.log(text);
        if (text == "") {
            start();
        }
    }
};

function init() {
    navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then(function(stream) {
        let settings = stream.getVideoTracks()[0].getSettings();
        width = PROCESSING_RESOLUTION_WIDTH;
        height = settings.height*PROCESSING_RESOLUTION_WIDTH/settings.width;
        
        tempCanvas.setAttribute("width", width);
        tempCanvas.setAttribute("height", height);
        outputCanvas.setAttribute("width", width);
        outputCanvas.setAttribute("height", height);

        video.setAttribute("width", width);
        video.setAttribute("height", height);
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.log(err);
    });
};

show_image = function (mat){
    let data = mat.data();
    let channels = mat.channels();
    let ctx = outputCanvas.getContext("2d");
    ctx.clearRect(0, 0, width, height);
    imdata = ctx.createImageData(mat.cols, mat.rows);
    for (var i = 0,j=0; i < data.length; i += channels, j+=4) {
        imdata.data[j] = data[i];
        imdata.data[j + 1] = data[i+1%channels];
        imdata.data[j + 2] = data[i+2%channels];
        imdata.data[j + 3] = 255;
    }
    ctx.putImageData(imdata, 0, 0);
}

function updateMask(x, y, w, h) {
    for (i=0; i < mask.data().length; i++) {
        mask.data()[i] = 0;
    }

    for (i=y; i < y + h; i++) {
        for (j=x; j < x + w; j++) {
            mask.data()[i*width + j] = 1;
        }
    }
}

function start() {
    streaming = true;
    startAndStopButton.style.display = "block";
    startAndStopButton.innerHTML = "Stop";
    colorButton.style.display = "block";

    let ctx = tempCanvas.getContext("2d");

    let src = new cv.Mat(height, width, cv.CV_8UC4);
    let dst1 = new cv.Mat(height, width, cv.CV_8UC3);
    let dst2 = new cv.Mat(height, width, cv.CV_8UC3);
    let orb = new cv.ORB(125, 1.2, 1, 0, 0, 2, 1, 20, 20);
    let keyPoints1 = new cv.KeyPointVector();
    let keyPoints2 = new cv.KeyPointVector();
    let descriptors1 = new cv.Mat();
    let sc = new cv.Scalar(0, 255, 0);
    let count = 0;
    mask = new cv.Mat(height, width, cv.CV_8UC1);

    function processVideo() {
        if (!streaming) {
            src.delete();
            dst1.delete();
            dst2.delete();
            orb.delete();
            keyPoints1.delete();
            keyPoints2.delete();
            descriptors1.delete();
            return;
        }

        try {
            let begin = performance.now();
            ctx.drawImage(video, 0, 0, width, height);
            src = cv.matFromArray(ctx.getImageData(0, 0, width, height), 24);
            // cv.resize(src, dst1, [outputCanvas.height, outputCanvas.width], 0, 0, 3);
            if (usingGray) {
                cv.cvtColor(src, dst1, cv.ColorConversionCodes.COLOR_RGBA2GRAY.value, 0);
            } else {
                cv.cvtColor(src, dst1, cv.ColorConversionCodes.COLOR_RGBA2RGB.value, 0);
            }
            if (detactFeatures) {
                orb.detect(dst1, keyPoints2, mask);
                // orb.compute(dst1, keyPoints2, descriptors1);
                cv.drawKeypoints(dst1, keyPoints2, dst2, sc, 0);
                show_image(dst2);
            } else {
                show_image(dst1);
            }

            let delta = performance.now() - begin;
            let delay = 1000/FPS - delta;
            count += Math.max(delta, 1000/FPS);
            if (count >= 1000) {
                document.getElementById('FPS').innerHTML = delay >= 0 ? "FPS: 30" : `FPS: ${Math.round(1000/delta)}`;
                count = 0;
            }
            setTimeout(processVideo, delay);
        } catch (err) {
            console.log(err);
        }
    }

    // schedule the first call.
    setTimeout(processVideo, 0);
}

startAndStopButton.addEventListener('click', () => {
    if (streaming) {
        streaming = false;
        startAndStopButton.innerHTML = "Start";
        outputCanvas.getContext('2d').clearRect(0, 0, width, height);
    } else {
        detactFeatures = false;
        start();
    }
});

colorButton.addEventListener('click', () => {
    usingGray = !usingGray;
    if (usingGray) {
        colorButton.innerHTML = "Color";
    } else {
        colorButton.innerHTML = "Gray";
    }
});

outputCanvas.addEventListener('click', (e) => {
    if (!streaming) {
        return;
    }

    let w = width/2;
    let h = height/2;
    let x = 0;
    let y = 0;

    if (e.layerX < w/2) {
        x = 0;
    } else if (e.layerX + w/2 > width) {
        x = width - w;
    } else {
        x = e.layerX - w/2;
    }

    if (e.layerY < h/2) {
        y = 0;
    } else if (e.layerY + h/2 > height) {
        y = height - h;
    } else {
        y = e.layerY - h/2;
    }

    updateMask(x, y, w, h);

    detactFeatures = true;
});


init();