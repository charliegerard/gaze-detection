import gaze from "./index.js";

const videoElement = document.querySelector("video");

const init = async () => {
  // set up webcam feed
  await gaze.setInputVideo(videoElement);

  // load model
  await gaze.loadModel();

  // get gaze detection

  const predict = async () => {
    let direction = await gaze.getGazePrediction();
    console.log("TEST", direction);
    let raf = requestAnimationFrame(predict);
  };
  predict();

  // stop detection
  // gaze.stop()
  // cancelAnimationFrame(raf)
};

init();
