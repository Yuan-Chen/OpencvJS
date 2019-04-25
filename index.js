// constants
const PROCESSING_RESOLUTION_WIDTH = 480;
const FPS = 30;
const AKAZE_THRESH = 3e-4;
const NN_MATCH_RATIO = 0.8;
const RANSAC_THRESH = 2.5;
const MIN_INLINER_RATIO = 0.25;
const MIN_INLINER_NUMBER = 20;

let width = 0;
let height = 0;
let streaming = false;
let cap;
let rect = {};
let drag = false;

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
let objectBB;

// load Opencv.js module
var Module = {
    setStatus: function(text) {
        console.log(text);
        if (text === "") {
            init();
        }
    }
};

function init() {
    // get camera
    navigator.mediaDevices.getUserMedia({video: true, audio: false})
    .then(function(stream) {
        let settings = stream.getVideoTracks()[0].getSettings();

        // adjust video width and height to match the requirement
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

function mouseDown(e) {
    // if clicked second image, do nothing
    if (e.layerY > height) {
        return;
    }

    // save the frame as a reference
    let src = new cv.Mat(height, width, cv.CV_8UC4);
    cap.read(src);
    cv.cvtColor(src, color, cv.COLOR_RGBA2RGB, 0);
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY, 0);
    src.delete();

    // save the coordinate of first click
    rect.x = e.layerX;
    rect.y = e.layerY;
    rect.w = 0;
    rect.h = 0;

    drag = true;
}

function mouseUp() {
    drag = false;

    // create an Mat for four vertices
    if (!objectBB.isDeleted()) {
        objectBB.delete();
    }
    objectBB = new cv.Mat(4, 1, cv.CV_32FC2);
    objectBB.data32F[0] = rect.x;
    objectBB.data32F[1] = rect.y;
    objectBB.data32F[2] = rect.x + rect.w;
    objectBB.data32F[3] = rect.y;
    objectBB.data32F[4] = rect.x + rect.w;
    objectBB.data32F[5] = rect.y + rect.h;
    objectBB.data32F[6] = rect.x;
    objectBB.data32F[7] = rect.y + rect.h;

    updateMask(mask, rect.x, rect.y, rect.w, rect.h);
    detector.detect(color, colorKeyPoints, mask);
    detector.detect(gray, grayKeyPoints, mask);
    detector.compute(color, colorKeyPoints, colorDescriptors);
    detector.compute(gray, grayKeyPoints, grayDescriptors);
    drawBoundingBox(color, objectBB);
    drawBoundingBox(gray, objectBB);

    detactFeatures = true;
}

function mouseMove(e) {
    if (drag) {
        rect.w = e.layerX - rect.x;
        if (e.layerY > height) {
            rect.h = height - rect.y;
        } else {
            rect.h = e.layerY - rect.y;
        }
    }
}

function start() {
    streaming = true;
    startAndStopButton.style.display = "block";
    startAndStopButton.innerHTML = "Stop";
    colorButton.style.display = "block";

    // first frame
    color = new cv.Mat(height, width, cv.CV_8UC3);
    colorKeyPoints = new cv.KeyPointVector();
    colorDescriptors = new cv.Mat();
    gray = new cv.Mat(height, width, cv.CV_8UC1);
    grayKeyPoints = new cv.KeyPointVector();
    grayDescriptors = new cv.Mat();

    // other frames
    let color2 = new cv.Mat(height, width, cv.CV_8UC3);
    let colorMatchingImage = new cv.Mat(2*height, width, cv.CV_8UC3);
    let gray2 = new cv.Mat(height, width, cv.CV_8UC1);
    let grayMatchingImage = new cv.Mat(2*height, width, cv.CV_8UC1);
    let newKeyPoints = new cv.KeyPointVector();
    let newDescriptors = new cv.Mat();

    // other variables
    cap = new cv.VideoCapture(video);
    // detector = new cv.ORB(500, 1.2, 1, 0);
    detector = new cv.AKAZE();
    detector.setThreshold(1e-3);
    let src = new cv.Mat(height, width, cv.CV_8UC4);
    let sc = new cv.Scalar(0, 255, 0, 0);
    mask = new cv.Mat(height, width, cv.CV_8UC1);
    let matcher = new cv.BFMatcher(2, false);
    let homography = new cv.Mat();
    let inlierMask = new cv.Mat();
    let newBB = new cv.Mat();
    let count = 0;
    objectBB = new cv.Mat();

    let originalFrame;
    let originalKeyPoints;
    let originalDescriptors;
    let newFrame;

    // add canvas event listeners
    outputCanvas.addEventListener('mousedown', mouseDown, false);
    outputCanvas.addEventListener('mouseup', mouseUp, false);
    outputCanvas.addEventListener('mousemove', mouseMove, false);

    function processVideo() {
        if (!streaming) {
            // remove canvas event listeners
            outputCanvas.removeEventListener('mousedown', mouseDown, false);
            outputCanvas.removeEventListener('mouseup', mouseUp, false);
            outputCanvas.removeEventListener('mousemove', mouseMove, false);

            // remove onjects
            color.delete();
            colorKeyPoints.delete();
            colorDescriptors.delete();
            gray.delete();
            grayKeyPoints.delete();
            grayDescriptors.delete();

            color2.delete();
            colorMatchingImage.delete();
            gray2.delete();
            grayMatchingImage.delete();
            newKeyPoints.delete();
            newDescriptors.delete();

            src.delete();
            detector.delete();
            mask.delete();
            objectBB.delete();

            matcher.delete();
            homography.delete();
            inlierMask.delete();

            return;
        }

        try {
            let begin = performance.now();
            cap.read(src);
            if (usingGray) {
                cv.cvtColor(src, gray2, cv.COLOR_RGBA2GRAY, 0);
                originalFrame = gray;
                originalKeyPoints = grayKeyPoints;
                originalDescriptors = grayDescriptors;
                newFrame = gray2;
            } else {
                cv.cvtColor(src, color2, cv.COLOR_RGBA2RGB, 0);
                originalFrame = color;
                originalKeyPoints = colorKeyPoints;
                originalDescriptors = colorDescriptors;
                newFrame = color2;
            }

            if (drag) {
                newFrame.data.set(originalFrame.data);
                cv.rectangle(newFrame, {x: rect.x, y:rect.y}, {x: rect.x+rect.w, y: rect.y+rect.h}, [255, 0, 0, 255]);
                cv.imshow("outputCanvas", newFrame);
            } else if (detactFeatures) {
                let matches = new cv.DMatchVectorVector();
                let goodMatches = new cv.DMatchVector();
                let inliers1 = new cv.KeyPointVector();
                let inliers2 = new cv.KeyPointVector();
                let inlierMatches = new cv.DMatchVector();

                detector.detect(newFrame, newKeyPoints);
                detector.compute(newFrame, newKeyPoints, newDescriptors);
                matcher.knnMatch(originalDescriptors, newDescriptors, matches, 2);
                for (let i = 0; i < matches.size(); i++) {
                    if (matches.get(i).size() < 2) {
                        continue;
                    }
                    let m = matches.get(i).get(0);
                    let n = matches.get(i).get(1);
                    if (m.distance < NN_MATCH_RATIO * n.distance) {
                        goodMatches.push_back(m);
                    }
                }

                let matched1 = new cv.Mat(goodMatches.size(), 1, cv.CV_32FC2);
                let matched2 = new cv.Mat(goodMatches.size(), 1, cv.CV_32FC2);
                for (let i = 0; i < goodMatches.size(); i++) {
                    matched1.data32F[2 * i] = originalKeyPoints.get(goodMatches.get(i).queryIdx).pt.x;
                    matched1.data32F[2 * i + 1] = originalKeyPoints.get(goodMatches.get(i).queryIdx).pt.y;
                    matched2.data32F[2 * i] = newKeyPoints.get(goodMatches.get(i).trainIdx).pt.x;
                    matched2.data32F[2 * i + 1] = newKeyPoints.get(goodMatches.get(i).trainIdx).pt.y;
                }

                if (goodMatches.size() >= 4) {
                    homography = cv.findHomography(matched1, matched2, cv.RANSAC, RANSAC_THRESH, inlierMask);

                    if (!homography.empty()) {
                        for (let i = 0; i < goodMatches.size(); i++) {
                            if (inlierMask.charAt(i) == 1) {
                                inlierMatches.push_back(goodMatches.get(i));
                            }
                        }

                        cv.perspectiveTransform(objectBB, newBB, homography);
                        if (inlierMatches.size() / matches.size() >= MIN_INLINER_RATIO || inlierMatches.size() >= MIN_INLINER_NUMBER) {
                            drawBoundingBox(newFrame, newBB);
                        }
                    }
                }

                if (usingGray) {
                    drawMatches(originalFrame, originalKeyPoints, newFrame, newKeyPoints, inlierMatches, grayMatchingImage, sc);
                    cv.imshow("outputCanvas", grayMatchingImage);
                } else {
                    drawMatches(originalFrame, originalKeyPoints, newFrame, newKeyPoints, inlierMatches, colorMatchingImage, sc);
                    cv.imshow("outputCanvas", colorMatchingImage);
                }
                // Image dimensions
                let w = newFrame.cols;
                let h = newFrame.rows;

                // Normalized dimensions:
                let maxSize = Math.max(w,h);
                let unitW = w / maxSize;
                let unitH = h / maxSize;

                let points2d = new cv.Mat(4, 1, cv.CV_32FC2);
                for (let i = 0; i < points2d.data32F.length; i++) {
                    points2d.data32F[i] = 0;
                }
                points2d.data32F[2] = w;
                points2d.data32F[4] = w;
                points2d.data32F[5] = h;
                points2d.data32F[7] = h;

                let points3d = new cv.Mat(4, 1, cv.CV_32FC3);
                for (let i = 0; i < points3d.data32F.length; i++) {
                    points3d.data32F[i] = 0;
                }
                points3d.data32F[0] = -unitW;
                points3d.data32F[1] = -unitH;
                points3d.data32F[3] = unitW;
                points3d.data32F[4] = -unitH;
                points3d.data32F[7] = unitW;
                points3d.data32F[8] = unitH;
                points3d.data32F[10] = -unitW;
                points3d.data32F[11] = unitH;

                let m_intrinsic = new cv.Mat(3, 3, cv.CV_32FC1);
                for (let i = 0; i < m_intrinsic.data32F.length; i++) {
                    m_intrinsic.data32F[i] = 0;
                }
                m_intrinsic.data32F[4] = 526.58037684199849;
                m_intrinsic.data32F[0] = 524.65577209994706;
                m_intrinsic.data32F[3] = 318.41744018680112;
                m_intrinsic.data32F[5] = 202.96659047014398;
                m_intrinsic.data32F[8] = 1;

                let m_distortion = new cv.Mat(5, 1, cv.CV_32FC1);
                for (let i = 0; i < 5; i++) {
                    m_distortion.data32F[i] = 0;
                }

                let rvec = new cv.Mat();
                let tvec = new cv.Mat();
                cv.solvePnP(points3d, points2d, m_intrinsic, m_distortion, rvec, tvec);

                matched1.delete();
                matched2.delete();
                goodMatches.delete();
                inliers1.delete();
                inliers2.delete();
                inlierMatches.delete();
            } else {
                cv.imshow("outputCanvas", newFrame);
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

// toggle start and stop
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

// toggle color and gray mode
colorButton.addEventListener('click', () => {
    usingGray = !usingGray;
    if (usingGray) {
        colorButton.innerHTML = "Color";
    } else {
        colorButton.innerHTML = "Gray";
    }
});