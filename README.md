# Gaze-detection - Detect gaze direction in JavaScript using Tensorflow.js

Use machine learning in JavaScript to detect eye movements and build gaze-controlled experiences!

## Demo

Visit [https://gaze-keyboard.netlify.app/](https://gaze-keyboard.netlify.app/) _(Works well on mobile too!!)_ 😃

![](gaze-demo.gif)

_Inspired by the Android application ["Look to speak"](https://play.google.com/store/apps/details?id=com.androidexperiments.looktospeak)._

Uses Tensorflow.js's [face landmark detection model](https://www.npmjs.com/package/@tensorflow-models/face-landmarks-detection).

## Detection

This tool detects when the user looks right, left, up and straight forward.

## How to use

### Install

As a npm module:

```bash
npm install gaze-detection --save
```

### Code sample

Start by importing it:

```js
import gaze from "gaze-detection";
```

The module needs a webcam feed to run the detection:

```js
const videoElement = document.querySelector("video");

const init = async () => {
  // set up webcam feed
  await gaze.setInputVideo(videoElement);
};

init();
```

Once the video stream is set up, load the model:

```js
await gaze.loadModel();
```

Run the predictions:

```js
const predict = async () => {
  const gazePrediction = await gaze.getGazePrediction();
  console.log("Gaze direction: ", gazePrediction); //will return 'RIGHT', 'LEFT', 'STRAIGHT' or 'TOP'
  if (gazePrediction === "RIGHT") {
    // do something when the user looks to the right
  }
  let raf = requestAnimationFrame(predict);
};
predict();
```

Stop the detection:

```js
cancelAnimationFrame(raf);
```
