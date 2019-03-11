const PROCESSING_RESOLUTION_WIDTH = 480;
const FPS = 30;

let width = 0;
let height = 0;
let streaming = false;
let cap;

let video = document.querySelector("#videoInput");
let tempCanvas = document.querySelector("#tempCanvas");
let outputCanvas = document.querySelector("#outputCanvas");
let startAndStopButton = document.querySelector('#startAndStop');
let colorButton = document.querySelector('#color');
let usingGray = false;
let detactFeatures = false;

let color;
let gray;
let colorKeyPoints;
let grayKeyPoints
let detector;
let colorDescriptors;
let grayDescriptors;
let mask;

var Module = {
    setStatus: function(text) {
        console.log(text);
        if (text === "") {
            init();
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

        start();
    })
    .catch(function(err) {
        console.log(err);
    });
};

 function show_image(mat){
    let data = mat.data;
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

function drawKeypoints(img, keyPoints, color) {
    for (i = 0 ; i < keyPoints.size() ; i++) {
        cv.circle(img, keyPoints.get(i).pt, 0, color);
        cv.circle(img, keyPoints.get(i).pt, 3, color);
    }
}

function drawMatches(img, keyPoints, img2, keyPoints2, matches, matchingImage, color) {
    drawKeypoints(img, keyPoints, color);
    matchingImage.data.set(img2.data);
    matchingImage.data.set(img.data, img2.data.length);
    for (i = 0; i < matches.size(); i++) {
        let m = keyPoints2.get(matches.get(i).trainIdx).pt;
        let n = keyPoints.get(matches.get(i).queryIdx).pt;
        n = {x: n.x, y: n.y + img.rows};
        cv.circle(matchingImage, m, 0, color);
        cv.circle(matchingImage, m, 3, color);
        cv.line(matchingImage, m, n, color);
    }
}

function start() {
    streaming = true;
    startAndStopButton.style.display = "block";
    startAndStopButton.innerHTML = "Stop";
    colorButton.style.display = "block";

    let ctx = tempCanvas.getContext("2d");

    // first frame
    color = new cv.Mat(height, width, cv.CV_8UC3);
    colorKeyPoints = new cv.KeyPointVector();
    colorDescriptors = new cv.Mat();
    gray = new cv.Mat(height, width, cv.CV_8UC1);
    grayKeyPoints = new cv.KeyPointVector();
    grayDescriptors = new cv.Mat();


    // other frames
    let color2 = new cv.Mat(height, width, cv.CV_8UC3);
    let colorKeyPoints2 = new cv.KeyPointVector();
    let colorDescriptors2 = new cv.Mat();
    let colorMatchingImage = new cv.Mat(2*height, width, cv.CV_8UC3);
    let gray2 = new cv.Mat(height, width, cv.CV_8UC1);
    let grayKeyPoints2 = new cv.KeyPointVector();
    let grayDescriptors2 = new cv.Mat();
    let grayMatchingImage = new cv.Mat(2*height, width, cv.CV_8UC1);

    // other variables
    cap = new cv.VideoCapture(video);
    // detector = new cv.ORB(500, 1.2, 1, 0);
    detector = new cv.AKAZE();
    detector.setThreshold(1e-3);
    let src = new cv.Mat(height, width, cv.CV_8UC4);
    let sc = new cv.Scalar(0, 255, 0, 0);
    mask = new cv.Mat(height, width, cv.CV_8UC1);
    let matcher = new cv.BFMatcher(2, false);
    const ratio = 0.8;
    let count = 0;

    function processVideo() {
        if (!streaming) {
            color.delete();
            colorKeyPoints.delete();
            colorDescriptors.delete();
            gray.delete();
            grayKeyPoints.delete();
            grayDescriptors.delete();

            color2.delete();
            colorKeyPoints2.delete();
            colorDescriptors2.delete();
            colorMatchingImage.delete();
            gray2.delete();
            grayKeyPoints2.delete();
            grayDescriptors2.delete();
            grayMatchingImage.delete();

            src.delete();
            detector.delete();
            mask.delete();

            matcher.delete();
            return;
        }

        try {
            let begin = performance.now();
            cap.read(src);
            if (detactFeatures) {
                let matches = new cv.DMatchVectorVector();
                let goodMatches = new cv.DMatchVector();

                if (usingGray) {
                    cv.cvtColor(src, gray2, cv.COLOR_RGBA2GRAY, 0);
                    detector.detect(gray2, grayKeyPoints2);
                    detector.compute(gray2, grayKeyPoints2, grayDescriptors2);
                    matcher.knnMatch(grayDescriptors, grayDescriptors2, matches, 2);
                    // drawKeypoints(gray2, grayKeyPoints2, sc);
                    for (let i = 0; i < matches.size(); i++) {
                        if (matches.get(i).size() < 2) {
                            continue;
                        }
                        let m = matches.get(i).get(0);
                        let n = matches.get(i).get(1);
                        if (m.distance < ratio * n.distance) {
                            goodMatches.push_back(m);
                        }
                    }
                    drawMatches(gray, grayKeyPoints, gray2, grayKeyPoints2, goodMatches, grayMatchingImage, sc);
                    cv.imshow("outputCanvas", grayMatchingImage);
                } else {
                    cv.cvtColor(src, color2, cv.COLOR_RGBA2RGB, 0);
                    detector.detect(color2, colorKeyPoints2);
                    detector.compute(color2, colorKeyPoints2, colorDescriptors2);
                    matcher.knnMatch(colorDescriptors, colorDescriptors2, matches, 2);
                    // drawKeypoints(color2, colorKeyPoints2, sc);
                    for (let i = 0; i < matches.size(); i++) {
                        if (matches.get(i).size() < 2) {
                            continue;
                        }
                        let m = matches.get(i).get(0);
                        let n = matches.get(i).get(1);
                        if (m.distance < ratio * n.distance) {
                            goodMatches.push_back(m);
                        }
                    }
                    drawMatches(color, colorKeyPoints, color2, colorKeyPoints2, goodMatches, colorMatchingImage, sc);
                    cv.imshow("outputCanvas", colorMatchingImage);
                }

                matches.delete();
                goodMatches.delete();
            } else {
                cv.imshow("outputCanvas", src);
            }

            let delta = performance.now() - begin;
            let delay = 1000/FPS - delta;
            count += Math.max(delta, 1000/FPS);
            if (count >= 1000) {
                document.getElementById('FPS').innerHTML = delay >= 0 ? "FPS: 30" : `FPS: ${Math.round(100000/delta)/100}`;
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
        outputCanvas.getContext('2d').clearRect(0, 0, outputCanvas.width, outputCanvas.height);
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

function updateMask(x, y, w, h) {
    for (i=0; i < mask.data.length; i++) {
        mask.data[i] = 0;
    }

    for (i=y; i < y + h; i++) {
        for (j=x; j < x + w; j++) {
            mask.data[i*width + j] = 1;
        }
    }
}

outputCanvas.addEventListener('click', (e) => {
    if (!streaming || e.layerY > height) {
        return;
    }

    let w = width/4;
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

    let src = new cv.Mat(height, width, cv.CV_8UC4);
    cap.read(src);

    cv.cvtColor(src, color, cv.COLOR_RGBA2RGB, 0);
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);

    updateMask(x, y, w, h);
    detector.detect(color, colorKeyPoints, mask);
    detector.detect(gray, grayKeyPoints, mask);
    detector.compute(color, colorKeyPoints, colorDescriptors);
    detector.compute(gray, grayKeyPoints, grayDescriptors);
    cv.rectangle(color, {x:x, y:y}, {x:x+w, y:y+h}, [255, 0, 0, 255]);
    cv.rectangle(gray, {x:x, y:y}, {x:x+w, y:y+h}, [255, 0, 0, 255]);

    detactFeatures = true;

    src.delete();
});