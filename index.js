import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";

let model, videoWidth, videoHeight, video, rafID;
let amountStraightEvents = 0;
const VIDEO_SIZE = 500;

async function setupCamera(videoElement) {
  video = videoElement;

  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      facingMode: "user",
      width: VIDEO_SIZE,
      height: VIDEO_SIZE,
    },
  });
  video.srcObject = stream;

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

const normalize = (val, max, min) =>
  Math.max(0, Math.min(1, (val - min) / (max - min)));

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
    return true;
  } else if (
    widthLeftSideFace > widthRightSideFace &&
    Math.abs(difference) > 5
  ) {
    return true;
  }
  return false;
};

async function renderPrediction() {
  const predictions = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: true,
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
        const positionLeftIrisX = video.width - positionXLeftIris;
        const normalizedXIrisPosition = normalize(
          positionLeftIrisX,
          faceTopRightX,
          faceBottomLeftX
        );

        if (normalizedXIrisPosition > 0.355) {
          xEvent = "RIGHT";
          //   if (previousXEvent !== xEvent) {
          //     previousXEvent = xEvent;
          //   }
        } else if (normalizedXIrisPosition < 0.315) {
          xEvent = "LEFT";
          //   if (previousXEvent !== xEvent) {
          //     previousXEvent = xEvent;
          //   }
        } else {
          amountStraightEvents++;
          if (amountStraightEvents > 10) {
            xEvent = "STRAIGHT";
            // previousXEvent = xEvent;
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

  //   rafID = requestAnimationFrame(renderPrediction);
  return xEvent;
}

async function main() {
  await tf.setBackend("webgl");

  video.play();
  videoWidth = video.videoWidth;
  videoHeight = video.videoHeight;
  video.width = videoWidth;
  video.height = videoHeight;

  model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    { maxFaces: 1 }
  );
}

const stopDetection = () => cancelAnimationFrame(rafID);

const gaze = {
  setInputVideo: setupCamera,
  loadModel: main,
  getGazePrediction: renderPrediction,
  stop: stopDetection,
};

module.exports = gaze;
