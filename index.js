import * as faceLandmarksDetection from "@tensorflow-models/face-landmarks-detection";
import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import * as tfjsWasm from "@tensorflow/tfjs-backend-wasm";
import { TRIANGULATION } from "./triangulation";
tfjsWasm.setWasmPaths(
  `https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-backend-wasm@${tfjsWasm.version_wasm}/dist/`
);

const NUM_KEYPOINTS = 468;
const NUM_IRIS_KEYPOINTS = 5;
const GREEN = "#32EEDB";
const RED = "#FF2C35";

function isMobile() {
  const isAndroid = /Android/i.test(navigator.userAgent);
  const isiOS = /iPhone|iPad|iPod/i.test(navigator.userAgent);
  return isAndroid || isiOS;
}

function distance(a, b) {
  return Math.sqrt(Math.pow(a[0] - b[0], 2) + Math.pow(a[1] - b[1], 2));
}

function drawPath(ctx, points, closePath) {
  const region = new Path2D();
  region.moveTo(points[0][0], points[0][1]);
  for (let i = 1; i < points.length; i++) {
    const point = points[i];
    region.lineTo(point[0], point[1]);
  }

  if (closePath) {
    region.closePath();
  }
  ctx.stroke(region);
}

let model, ctx, videoWidth, videoHeight, video, canvas, rafID;

const VIDEO_SIZE = 500;
const mobile = isMobile();
// Don't render the point cloud on mobile in order to maximize performance and
// to avoid crowding limited screen space.

const state = {
  backend: "webgl",
  maxFaces: 1,
  triangulateMesh: true,
  predictIrises: true,
};

async function setupCamera() {
  video = document.getElementById("video");

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

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

let positionXLeftIris;
let previousXPositionLeftIris;
let positionYLeftIris;
let previousYPositionLeftIris;

const normalize = (val, max, min) =>
  Math.max(0, Math.min(1, (val - min) / (max - min)));

async function renderPrediction() {
  const predictions = await model.estimateFaces({
    input: video,
    returnTensors: false,
    flipHorizontal: false,
    predictIrises: state.predictIrises,
  });
  ctx.drawImage(
    video,
    0,
    0,
    videoWidth,
    videoHeight,
    0,
    0,
    canvas.width,
    canvas.height
  );

  if (predictions.length > 0) {
    predictions.forEach((prediction) => {
      const keypoints = prediction.scaledMesh;

      positionXLeftIris = prediction.annotations.leftEyeIris[0][0];
      positionYLeftIris = prediction.annotations.leftEyeIris[0][1];

      const faceBottomLeft =
        video.width - prediction.boundingBox.bottomRight[0]; // face is flipped horizontally so bottom right is actually bottom left.

      const faceTopRight = video.width - prediction.boundingBox.topLeft[0]; // face is flipped horizontally so top left is actually top right.

      if (faceBottomLeft > 0) {
        const faceWidth = faceTopRight - faceBottomLeft;

        // When faceWidth is 200px, position left iris is between -3 / + 3
        // when faceWidth is 300px, position left iris is between -5 / + 5

        if (previousXPositionLeftIris !== positionXLeftIris) {
          //   let difference = positionXLeftIris - previousXPositionLeftIris;
          const position = video.width - positionXLeftIris;
          const testIrisPosition = normalize(
            position,
            faceTopRight,
            faceBottomLeft
          );
          if (testIrisPosition > 0.35) {
            console.log("RIGHT");
          } else if (testIrisPosition < 0.31) {
            console.log("LEFT");
          } else {
            console.log("STRAIGHT");
          }
          //     if (difference > 0 && difference > 3) {
          //       console.log("RIGHT");
          //     } else if (difference < 0 && difference < -3) {
          //       console.log("LEFT");
          //     }
          //   previousXPositionLeftIris = positionXLeftIris;
        }
      }

      // Looking left/right
      //   if (previousXPositionLeftIris !== positionXLeftIris) {
      //     let difference = positionXLeftIris - previousXPositionLeftIris;
      //     if (difference > 0 && difference > 3) {
      //       console.log("RIGHT");
      //     } else if (difference < 0 && difference < -3) {
      //       console.log("LEFT");
      //     }
      //     previousXPositionLeftIris = positionXLeftIris;
      //   }

      // Looking top/down
      //   if (previousYPositionLeftIris !== positionYLeftIris) {
      //     let difference = positionYLeftIris - previousYPositionLeftIris;
      //     if (difference > 0 && difference > 2) {
      //       console.log("BOTTOM");
      //     } else if (difference < 0 && difference < -2) {
      //       console.log("TOP");
      //     }
      //     previousYPositionLeftIris = positionYLeftIris;
      //   }

      if (prediction.boundingBox.topLeft) {
        ctx.fillStyle = GREEN;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.arc(
          prediction.boundingBox.topLeft[0],
          prediction.boundingBox.topLeft[1],
          10 /* radius */,
          0,
          2 * Math.PI
        );
        ctx.fill();
      }

      if (prediction.boundingBox.bottomRight) {
        ctx.fillStyle = RED;
        ctx.lineWidth = 0.5;
        ctx.beginPath();
        ctx.arc(
          prediction.boundingBox.bottomRight[0],
          prediction.boundingBox.bottomRight[1],
          10 /* radius */,
          0,
          2 * Math.PI
        );
        ctx.fill();
      }

      if (state.triangulateMesh) {
        ctx.strokeStyle = GREEN;
        ctx.lineWidth = 0.5;

        for (let i = 0; i < TRIANGULATION.length / 3; i++) {
          const points = [
            TRIANGULATION[i * 3],
            TRIANGULATION[i * 3 + 1],
            TRIANGULATION[i * 3 + 2],
          ].map((index) => keypoints[index]);

          drawPath(ctx, points, true);
        }
      } else {
        ctx.fillStyle = GREEN;

        for (let i = 0; i < NUM_KEYPOINTS; i++) {
          const x = keypoints[i][0];
          const y = keypoints[i][1];

          ctx.beginPath();
          ctx.arc(x, y, 1 /* radius */, 0, 2 * Math.PI);
          ctx.fill();
        }
      }

      if (keypoints.length > NUM_KEYPOINTS) {
        ctx.strokeStyle = RED;
        ctx.lineWidth = 1;

        const leftCenter = keypoints[NUM_KEYPOINTS];
        const leftDiameterY = distance(
          keypoints[NUM_KEYPOINTS + 4],
          keypoints[NUM_KEYPOINTS + 2]
        );
        const leftDiameterX = distance(
          keypoints[NUM_KEYPOINTS + 3],
          keypoints[NUM_KEYPOINTS + 1]
        );

        ctx.beginPath();
        ctx.ellipse(
          leftCenter[0],
          leftCenter[1],
          leftDiameterX / 2,
          leftDiameterY / 2,
          0,
          0,
          2 * Math.PI
        );
        ctx.stroke();

        if (keypoints.length > NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS) {
          const rightCenter = keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS];
          const rightDiameterY = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 2],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 4]
          );
          const rightDiameterX = distance(
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 3],
            keypoints[NUM_KEYPOINTS + NUM_IRIS_KEYPOINTS + 1]
          );

          ctx.beginPath();
          ctx.ellipse(
            rightCenter[0],
            rightCenter[1],
            rightDiameterX / 2,
            rightDiameterY / 2,
            0,
            0,
            2 * Math.PI
          );
          ctx.stroke();
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

  canvas = document.getElementById("output");
  canvas.width = videoWidth;
  canvas.height = videoHeight;
  const canvasContainer = document.querySelector(".canvas-wrapper");
  canvasContainer.style = `width: ${videoWidth}px; height: ${videoHeight}px`;

  ctx = canvas.getContext("2d");
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.fillStyle = GREEN;
  ctx.strokeStyle = GREEN;
  ctx.lineWidth = 0.5;

  model = await faceLandmarksDetection.load(
    faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
    { maxFaces: state.maxFaces }
  );
  renderPrediction();
}

main();
