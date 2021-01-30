import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
// import "@tensorflow/tfjs-backend-cpu";
// import * as tfjsWasm from "@tensorflow/tfjs-backend-wasm";
// tfjsWasm.setWasmPaths(
//   `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`
// );

import {
  initialLetters,
  commands,
  VIDEO_SIZE,
  isMobile,
  state,
  mobile,
  normalize,
  isEven,
} from "../utils";

let letters = initialLetters;

let amountStraightEvents = 0;

const leftSection = document.querySelector(".section-left");
const rightSection = document.querySelector(".section-right");

let model, videoWidth, videoHeight, video, rafID;

const initKeyboardUI = (lettersSelected, update) => {
  if (leftSection.hasChildNodes() && update) {
    leftSection.innerHTML = "";
  }
  if (rightSection.hasChildNodes() && update) {
    rightSection.innerHTML = "";
  }
  if (lettersSelected.length === 1) {
    const outputElement = document.querySelector(".output");
    if (lettersSelected[0] === "Delete") {
      outputElement.value = outputElement.value.slice(0, -1);
    } else if (lettersSelected[0] === "space") {
      outputElement.value += " ";
    } else if (lettersSelected[0] === "Enter") {
      outputElement.value += "</br>";
    } else {
      outputElement.value += lettersSelected[0];
    }

    initKeyboardUI(initialLetters, false);
    letterPrinted();
    return;
  } else {
    lettersSelected.map((letter, i) => {
      const letterElement = document.createElement("p");
      letterElement.innerHTML = letter;
      if (i < Math.round(lettersSelected.length / 2)) {
        leftSection.appendChild(letterElement);
      } else {
        rightSection.appendChild(letterElement);
      }
    });
  }
};

initKeyboardUI(letters, false);

const letterPrinted = () => (letters = initialLetters);

async function setupCamera() {
  video = document.getElementById("video");

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "user",
        // Only setting the video to a specified size in order to accommodate a
        // point cloud, so on mobile devices accept the default size.
        width: mobile ? undefined : VIDEO_SIZE,
        height: mobile ? undefined : VIDEO_SIZE,
      },
    });
    video.srcObject = stream;
  } catch (err) {
    throw new Error(
      "Looks like you've denied webcam access. You need to give the webcam permission to be able to use this demo.",
      err
    );
  }

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

let positionXLeftIris;
let positionYLeftIris;
let previousXEvent;
let xEvent;

const isFaceRotated = (landmarks) => {
  const leftCheek = landmarks.leftCheek;
  const rightCheek = landmarks.rightCheek;
  const midwayBetweenEyes = landmarks.midwayBetweenEyes;

  const xPositionLeftCheek = video.width - leftCheek[0][0];
  const xPositionRightCheek = video.width - rightCheek[0][0];
  const xPositionMidwayBetweenEyes = video.width - midwayBetweenEyes[0][0];

  const widthLeftSideFace = xPositionMidwayBetweenEyes - xPositionLeftCheek;
  const widthRightSideFace = xPositionRightCheek - xPositionMidwayBetweenEyes;

  const difference = widthRightSideFace - widthLeftSideFace;

  if (widthLeftSideFace < widthRightSideFace && Math.abs(difference) > 5) {
    console.log("ROTATING FACE LEFT");
    return true;
  } else if (
    widthLeftSideFace > widthRightSideFace &&
    Math.abs(difference) > 5
  ) {
    console.log("ROTATING FACE RIGHT");
    return true;
  }
  return false;
};

async function renderPrediction() {
  const predictions = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: state.predictIrises,
  });

  if (predictions.length > 0) {
    predictions.forEach((prediction) => {
      positionXLeftIris = prediction.annotations.leftEyeIris[0][0];
      positionYLeftIris = prediction.annotations.leftEyeIris[0][1];

      const faceBottomLeftX =
        video.width - prediction.boundingBox.bottomRight[0]; // face is flipped horizontally so bottom right is actually bottom left.
      const faceBottomLeftY = prediction.boundingBox.bottomRight[1];

      const faceTopRightX = video.width - prediction.boundingBox.topLeft[0]; // face is flipped horizontally so top left is actually top right.
      const faceTopRightY = prediction.boundingBox.topLeft[1];

      if (faceBottomLeftX > 0 && !isFaceRotated(prediction.annotations)) {
        // When faceWidth is 200px, position left iris is between -3 / + 3
        // when faceWidth is 300px, position left iris is between -5 / + 5
        const positionLeftIrisX = video.width - positionXLeftIris;
        const normalizedXIrisPosition = normalize(
          positionLeftIrisX,
          faceTopRightX,
          faceBottomLeftX
        );

        if (normalizedXIrisPosition > 0.355) {
          xEvent = "RIGHT";

          if (previousXEvent !== xEvent) {
            letters = letters.slice(
              Math.round(letters.length / 2),
              letters.length
            );
            initKeyboardUI(letters, true);
            previousXEvent = xEvent;
          }
        } else if (normalizedXIrisPosition < 0.315) {
          xEvent = "LEFT";

          if (previousXEvent !== xEvent) {
            letters = letters.slice(0, Math.round(letters.length / 2));
            initKeyboardUI(letters.slice(letters), true);
            previousXEvent = xEvent;
          }
        } else {
          amountStraightEvents++;
          if (amountStraightEvents > 10) {
            xEvent = "STRAIGHT";
            previousXEvent = xEvent;
            amountStraightEvents = 0;
          }
        }

        const normalizedYIrisPosition = normalize(
          positionYLeftIris,
          faceTopRightY,
          faceBottomLeftY
        );

        if (normalizedYIrisPosition > 0.63) {
          //   console.log("TOP");
        } else if (normalizedYIrisPosition < 0.615) {
          //   console.log("BOTTOM"); // meh doesn't reallyyyyy work
        } else {
          // console.log('STRAIGHT')
        }
      }
    });
  }

  rafID = requestAnimationFrame(renderPrediction);
}

async function main() {
  await tf.setBackend(state.backend);

  await setupCamera();
  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    { maxFaces: state.maxFaces }
  );
  renderPrediction();
}

main();
