import gaze from "./index.js";

const videoElement = document.querySelector("video");

const init = async () => {
  await gaze.setInputVideo(videoElement);
  await gaze.loadModel();

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
