// concat two images vertically and draw matched key points
function drawMatches(img, keyPoints, img2, keyPoints2, matches, outputImage, color) {
    outputImage.data.set(img2.data);
    outputImage.data.set(img.data, img2.data.length);
    for (i = 0; i < matches.size(); i++) {
        let m = keyPoints2.get(matches.get(i).trainIdx).pt;
        let n = keyPoints.get(matches.get(i).queryIdx).pt;
        n = {x: n.x, y: n.y + img.rows};
        cv.circle(outputImage, m, 0, color);
        cv.circle(outputImage, m, 3, color);
        cv.line(outputImage, m, n, color);
        cv.circle(outputImage, n, 0, color);
        cv.circle(outputImage, n, 3, color);
    }
}

// draw bounding rectangle for given four vertices
function drawBoundingBox(img, bb) {
    cv.line(img, {x: bb.data32F[0], y:bb.data32F[1]}, {x: bb.data32F[2], y:bb.data32F[3]}, [255, 0, 0, 255]);
    cv.line(img, {x: bb.data32F[2], y:bb.data32F[3]}, {x: bb.data32F[4], y:bb.data32F[5]}, [255, 0, 0, 255]);
    cv.line(img, {x: bb.data32F[4], y:bb.data32F[5]}, {x: bb.data32F[6], y:bb.data32F[7]}, [255, 0, 0, 255]);
    cv.line(img, {x: bb.data32F[6], y:bb.data32F[7]}, {x: bb.data32F[0], y:bb.data32F[1]}, [255, 0, 0, 255]);
}

// fill an mask Mat with 1 in the given rectangle and 0 elsewhere
function updateMask(mask, x, y, w, h) {
    for (i=0; i < mask.data.length; i++) {
        mask.data[i] = 0;
    }

    // revert width and height if they are negative
    if (w < 0) {
        x += w;
        w *= -1;
    }
    
    if (h < 0) {
        y += h;
        h *= -1;
    }

    for (i=y; i < y + h; i++) {
        for (j=x; j < x + w; j++) {
            mask.data[i*width + j] = 1;
        }
    }
}